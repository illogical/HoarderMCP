"""Document ingestion service for HoarderMCP."""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from pydantic import BaseModel, Field, HttpUrl

from ..models.document import Document, DocumentChunk, DocumentMetadata, DocumentType
from ..storage import VectorStore, VectorStoreFactory, VectorStoreConfig

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Language-specific splitters
LANGUAGE_SPLITTERS = {
    DocumentType.PYTHON: {
        "language": Language.PYTHON,
        "separators": [
            "\n\n\n",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    },
    DocumentType.CSHARP: {
        "language": Language.CSHARP,
        "separators": [
            "\n\n\n",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    },
    DocumentType.MARKDOWN: {
        "separators": [
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " ",
            "",
        ]
    },
}

# Default text splitter configuration
DEFAULT_SPLITTER_CONFIG = {
    "separators": ["\n\n", "\n", " ", ""],
    "keep_separator": True,
    "is_separator_regex": False,
}


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    chunk_size: int = Field(
        DEFAULT_CHUNK_SIZE,
        description="Maximum size of each chunk in characters",
        gt=0,
    )
    chunk_overlap: int = Field(
        DEFAULT_CHUNK_OVERLAP,
        description="Number of characters to overlap between chunks",
        ge=0,
    )
    length_function: str = Field(
        "len",
        description="Function to use for calculating text length (e.g., 'len' or 'token_counter')",
    )
    keep_separator: bool = Field(
        True,
        description="Whether to keep the separators in the chunks",
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
        """Split a document into chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of document chunks.
        """
        content_type = document.metadata.content_type
        content = document.content

        # Get splitter configuration
        splitter_config = LANGUAGE_SPLITTERS.get(content_type, DEFAULT_SPLITTER_CONFIG)

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunking_config.chunk_size,
            chunk_overlap=self.chunking_config.chunk_overlap,
            length_function=len,  # TODO: Support token counting
            **splitter_config,
        )

        # Split text
        texts = text_splitter.split_text(content)

        # Create chunks
        chunks = []
        for i, text in enumerate(texts):
            chunk_id = f"{document.id}_chunk_{i}"
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document.id,
                content=text,
                metadata=document.metadata.copy(),
                chunk_index=i,
                chunk_count=len(texts),
            )
            chunks.append(chunk)

        return chunks

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
