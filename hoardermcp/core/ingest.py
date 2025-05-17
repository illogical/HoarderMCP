"""Document ingestion service for HoarderMCP."""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.text_splitter import Language
from pydantic import BaseModel, Field, HttpUrl

from .chunking import ChunkingConfig, ChunkingFactory, ChunkingStrategy
from ..models.document import Document, DocumentChunk, DocumentMetadata, DocumentType
from ..storage import VectorStore, VectorStoreFactory, VectorStoreConfig

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CHUNKING_STRATEGY = ChunkingStrategy.SEMANTIC


class ChunkingConfig(ChunkingConfig):
    """Extended configuration for document chunking.
    
    This class extends the base ChunkingConfig to add additional fields
    specific to the ingestion service.
    """
    
    length_function: str = Field(
        "len",
        description="Function to use for calculating text length (e.g., 'len' or 'token_counter')",
    )
    add_start_index: bool = Field(
        False,
        description="Whether to add the start index of each chunk to its metadata",
    )
    
    class Config:
        """Pydantic config."""
        extra = "forbid"
        use_enum_values = True


class EmbeddingConfig(BaseModel):
    """Configuration for document embeddings."""

    model_name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the embedding model to use",
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the embedding model",
    )
    encode_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the model's encode method",
    )
    batch_size: int = Field(
        32,
        description="Number of texts to embed in each batch",
        gt=0,
    )
    show_progress: bool = Field(
        False,
        description="Whether to show a progress bar during embedding",
    )

    class Config:
        """Pydantic config."""

        extra = "forbid"


class IngestResult(BaseModel):
    """Result of a document ingestion operation."""

    document_id: str = Field(..., description="Unique identifier for the document")
    chunks: List[DocumentChunk] = Field(
        default_factory=list, description="Document chunks"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class IngestService:
    """Service for ingesting documents into the vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        chunking_config: Optional[ChunkingConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        """Initialize the ingestion service.

        Args:
            vector_store: Vector store to ingest documents into.
            chunking_config: Configuration for document chunking.
            embedding_config: Configuration for document embeddings.
        """
        self.vector_store = vector_store
        self.chunking_config = chunking_config or ChunkingConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self._embedding_model = None

    async def initialize(self) -> None:
        """Initialize the ingestion service."""
        if not self.vector_store.is_initialized:
            await self.vector_store.initialize()

    async def ingest_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        content_type: Optional[Union[str, DocumentType]] = None,
        **kwargs,
    ) -> IngestResult:
        """Ingest a single document.

        Args:
            content: Document content.
            metadata: Document metadata.
            content_type: Type of the document content.
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            IngestResult containing the document ID and chunks.
        """
        # Create document metadata
        doc_metadata = self._create_document_metadata(metadata or {}, content_type)

        # Create document
        document = Document(
            id=str(uuid.uuid4()),
            content=content,
            metadata=doc_metadata,
        )

        # Chunk document
        chunks = self._chunk_document(document)
        document.chunks = chunks

        # Generate embeddings
        if self.embedding_config:
            await self._generate_embeddings(chunks)

        # Store document chunks
        chunk_ids = await self.vector_store.add_chunks(chunks, **kwargs)

        return IngestResult(
            document_id=document.id,
            chunks=chunks,
            metadata={"chunk_ids": chunk_ids},
        )

    async def ingest_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        **kwargs,
    ) -> List[IngestResult]:
        """Ingest multiple documents.

        Args:
            documents: List of documents to ingest.
            batch_size: Number of documents to process in each batch.
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            List of IngestResult objects.
        """
        results = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[
                    self.ingest_document(
                        content=doc.content,
                        metadata=doc.metadata.dict(),
                        content_type=doc.metadata.content_type,
                        **kwargs,
                    )
                    for doc in batch
                ]
            )
            results.extend(batch_results)
        return results

    def _create_document_metadata(
        self, metadata: Dict[str, Any], content_type: Optional[Union[str, DocumentType]]
    ) -> DocumentMetadata:
        """Create document metadata.

        Args:
            metadata: Raw metadata dictionary.
            content_type: Type of the document content.

        Returns:
            DocumentMetadata instance.
        """
        # Determine content type
        if isinstance(content_type, str):
            try:
                content_type = DocumentType(content_type.lower())
            except ValueError:
                content_type = DocumentType.TEXT
        elif content_type is None:
            content_type = DocumentType.TEXT

        # Create metadata
        return DocumentMetadata(
            content_type=content_type,
            **{
                k: v
                for k, v in metadata.items()
                if k in DocumentMetadata.__annotations__
            },
        )

    def _chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a document into chunks using the configured strategy.

        Args:
            document: Document to chunk.

        Returns:
            List of document chunks.
        """
        # Create chunker based on document type and configuration
        chunker = ChunkingFactory.get_chunker(
            doc_type=document.metadata.content_type,
            config=ChunkingConfig(
                strategy=ChunkingStrategy.SEMANTIC,
                chunk_size=self.chunking_config.chunk_size,
                chunk_overlap=self.chunking_config.chunk_overlap,
                max_tokens=self.chunking_config.max_tokens,
                language={
                    DocumentType.PYTHON: Language.PYTHON,
                    DocumentType.CSHARP: Language.CSHARP,
                }.get(document.metadata.content_type),
            ),
        )
        
        # Split document into chunks
        return chunker.chunk_document(document)

    async def _generate_embeddings(
        self, chunks: List[DocumentChunk], **kwargs
    ) -> None:
        """Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks.
            **kwargs: Additional arguments to pass to the embedding model.
        """
        if not chunks:
            return

        # Lazy load embedding model
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(
                    self.embedding_config.model_name,
                    **self.embedding_config.model_kwargs,
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. Install with: pip install sentence-transformers"
                )
                return

        # Batch process chunks
        batch_size = self.embedding_config.batch_size
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk.content for chunk in batch]

            # Generate embeddings
            embeddings = self._embedding_model.encode(
                texts, **self.embedding_config.encode_kwargs
            )

            # Assign embeddings to chunks
            for j, embedding in enumerate(embeddings):
                batch[j].vector_embedding = embedding.tolist()

    async def close(self) -> None:
        """Close the ingestion service and release resources."""
        if self.vector_store:
            await self.vector_store.close()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
