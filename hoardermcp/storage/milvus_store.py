"""Milvus vector store implementation."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
    MilvusException,
)

from ..models.document import Document, DocumentChunk, DocumentType
from .base import VectorStore, VectorStoreConfig, VectorStoreType

logger = logging.getLogger(__name__)

# Default collection name
DEFAULT_COLLECTION_NAME = "hoardermcp_documents"

# Field names
ID_FIELD = "id"
DOC_ID_FIELD = "doc_id"
CHUNK_ID_FIELD = "chunk_id"
CONTENT_FIELD = "content"
VECTOR_FIELD = "vector"
METADATA_FIELD = "metadata"
CONTENT_TYPE_FIELD = "content_type"
EMBEDDING_MODEL_FIELD = "embedding_model"

# Index parameters
DEFAULT_INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024},
}

# Search parameters
DEFAULT_SEARCH_PARAMS = {"metric_type": "L2", "params": {"nprobe": 16}}


class MilvusConfig(VectorStoreConfig):
    """Configuration for Milvus vector store."""

    type: VectorStoreType = Field(
        VectorStoreType.MILVUS, const=True, description="Type of vector store"
    )
    host: str = Field("localhost", description="Milvus server host")
    port: int = Field(19530, description="Milvus server port")
    user: str = Field("", description="Username for authentication")
    password: str = Field("", description="Password for authentication")
    db_name: str = Field("default", description="Database name")
    collection_name: str = Field(
        DEFAULT_COLLECTION_NAME, description="Collection name"
    )
    auto_id: bool = Field(True, description="Whether to auto-generate document IDs")
    drop_old: bool = Field(
        False, description="Whether to drop existing collection if it exists"
    )
    index_params: Dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_INDEX_PARAMS.copy(),
        description="Index parameters",
    )
    search_params: Dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_SEARCH_PARAMS.copy(),
        description="Search parameters",
    )
    consistency_level: str = Field(
        "Session",
        description="Consistency level (Strong, Session, Bounded, Eventually)",
    )
    use_alias: bool = Field(
        False, description="Whether to use collection alias instead of name"
    )

    class Config:
        """Pydantic config."""

        extra = "forbid"
        use_enum_values = True


class MilvusVectorStore(VectorStore[MilvusConfig]):
    """Milvus vector store implementation."""

    def __init__(self, config: MilvusConfig):
        """Initialize the Milvus vector store.

        Args:
            config: Configuration for the Milvus vector store.
        """
        super().__init__(config)
        self._collection: Optional[Collection] = None
        self._connection_alias = f"hoardermcp_{id(self)}"

    async def initialize(self) -> None:
        """Initialize the Milvus vector store."""
        if self._initialized:
            return

        try:
            # Connect to Milvus server
            connections.connect(
                alias=self._connection_alias,
                host=self.config.host,
                port=self.config.port,
                user=self.config.user or None,
                password=self.config.password or None,
                db_name=self.config.db_name or "default",
            )

            # Create collection if it doesn't exist
            if not utility.has_collection(
                self.config.collection_name, using=self._connection_alias
            ):
                self._create_collection()
            else:
                if self.config.drop_old:
                    utility.drop_collection(
                        self.config.collection_name, using=self._connection_alias
                    )
                    self._create_collection()
                else:
                    self._collection = Collection(
                        self.config.collection_name,
                        using=self._connection_alias,
                    )

            # Load collection for search
            self._collection.load()
            self._initialized = True
            logger.info(
                f"Milvus vector store initialized with collection '{self.config.collection_name}'"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Milvus vector store: {e}")
            raise

    def _create_collection(self) -> None:
        """Create a new collection with the specified schema."""
        # Define fields
        fields = [
            FieldSchema(
                name=ID_FIELD,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name=DOC_ID_FIELD,
                dtype=DataType.VARCHAR,
                max_length=255,
                description="Document ID",
            ),
            FieldSchema(
                name=CHUNK_ID_FIELD,
                dtype=DataType.VARCHAR,
                max_length=255,
                description="Chunk ID",
            ),
            FieldSchema(
                name=CONTENT_FIELD,
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Content text",
            ),
            FieldSchema(
                name=CONTENT_TYPE_FIELD,
                dtype=DataType.VARCHAR,
                max_length=50,
                description="Content type (e.g., 'markdown', 'python')",
            ),
            FieldSchema(
                name=EMBEDDING_MODEL_FIELD,
                dtype=DataType.VARCHAR,
                max_length=255,
                description="Name of the embedding model used",
            ),
            FieldSchema(
                name=VECTOR_FIELD,
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.dimension,
                description="Vector embedding",
            ),
            FieldSchema(
                name=METADATA_FIELD,
                dtype=DataType.JSON,
                description="Document metadata",
            ),
        ]

        # Create schema
        schema = CollectionSchema(
            fields=fields,
            description="Document chunks with vector embeddings",
            enable_dynamic_field=True,
        )

        # Create collection
        self._collection = Collection(
            name=self.config.collection_name,
            schema=schema,
            using=self._connection_alias,
            consistency_level=self.config.consistency_level,
        )

        # Create index on the vector field
        self._collection.create_index(
            field_name=VECTOR_FIELD,
            index_params=self.config.index_params,
        )

        logger.info(
            f"Created new collection '{self.config.collection_name}' with vector index"
        )

    async def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        **kwargs,
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.
            batch_size: Number of documents to add in each batch.
            **kwargs: Additional arguments.

        Returns:
            List of document IDs that were added.
        """
        if not self._initialized:
            await self.initialize()

        doc_ids = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_doc_ids = await self._add_documents_batch(batch, **kwargs)
            doc_ids.extend(batch_doc_ids)

        return doc_ids

    async def _add_documents_batch(
        self, documents: List[Document], **kwargs
    ) -> List[str]:
        """Add a batch of documents to the vector store.

        Args:
            documents: List of documents to add.
            **kwargs: Additional arguments.

        Returns:
            List of document IDs that were added.
        """
        if not documents:
            return []

        # Prepare data for insertion
        data = []
        doc_ids = []

        for doc in documents:
            doc_id = doc.id or str(uuid.uuid4())
            doc_ids.append(doc_id)

            # Add each chunk as a separate row
            for chunk in doc.chunks or []:
                chunk_id = chunk.id or str(uuid.uuid4())
                vector = chunk.vector_embedding or []

                if len(vector) != self.config.dimension:
                    logger.warning(
                        f"Skipping chunk {chunk_id} due to invalid vector dimension: "
                        f"expected {self.config.dimension}, got {len(vector)}"
                    )
                    continue

                data.append(
                    {
                        DOC_ID_FIELD: doc_id,
                        CHUNK_ID_FIELD: chunk_id,
                        CONTENT_FIELD: chunk.content,
                        CONTENT_TYPE_FIELD: doc.metadata.content_type.value,
                        EMBEDDING_MODEL_FIELD: kwargs.get("embedding_model", "unknown"),
                        VECTOR_FIELD: vector,
                        METADATA_FIELD: {
                            **doc.metadata.dict(),
                            "chunk_index": chunk.chunk_index,
                            "chunk_count": chunk.chunk_count,
                        },
                    }
                )

        # Insert data into Milvus
        if data:
            try:
                result = self._collection.insert(data)
                logger.info(f"Inserted {len(data)} chunks into Milvus")
                return doc_ids
            except Exception as e:
                logger.error(f"Failed to insert documents into Milvus: {e}")
                raise

        return doc_ids

    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100,
        **kwargs,
    ) -> List[str]:
        """Add document chunks to the vector store.

        Args:
            chunks: List of document chunks to add.
            batch_size: Number of chunks to add in each batch.
            **kwargs: Additional arguments.

        Returns:
            List of chunk IDs that were added.
        """
        if not self._initialized:
            await self.initialize()

        chunk_ids = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_chunk_ids = await self._add_chunks_batch(batch, **kwargs)
            chunk_ids.extend(batch_chunk_ids)

        return chunk_ids

    async def _add_chunks_batch(
        self, chunks: List[DocumentChunk], **kwargs
    ) -> List[str]:
        """Add a batch of document chunks to the vector store.

        Args:
            chunks: List of document chunks to add.
            **kwargs: Additional arguments.

        Returns:
            List of chunk IDs that were added.
        """
        if not chunks:
            return []

        # Prepare data for insertion
        data = []
        chunk_ids = []

        for chunk in chunks:
            chunk_id = chunk.id or str(uuid.uuid4())
            vector = chunk.vector_embedding or []

            if len(vector) != self.config.dimension:
                logger.warning(
                    f"Skipping chunk {chunk_id} due to invalid vector dimension: "
                    f"expected {self.config.dimension}, got {len(vector)}"
                )
                continue

            data.append(
                {
                    DOC_ID_FIELD: chunk.document_id or "",
                    CHUNK_ID_FIELD: chunk_id,
                    CONTENT_FIELD: chunk.content,
                    CONTENT_TYPE_FIELD: chunk.metadata.content_type.value,
                    EMBEDDING_MODEL_FIELD: kwargs.get("embedding_model", "unknown"),
                    VECTOR_FIELD: vector,
                    METADATA_FIELD: {
                        **chunk.metadata.dict(),
                        "chunk_index": chunk.chunk_index,
                        "chunk_count": chunk.chunk_count,
                    },
                }
            )
            chunk_ids.append(chunk_id)

        # Insert data into Milvus
        if data:
            try:
                self._collection.insert(data)
                logger.info(f"Inserted {len(data)} chunks into Milvus")
            except Exception as e:
                logger.error(f"Failed to insert chunks into Milvus: {e}")
                raise

        return chunk_ids

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[DocumentChunk]:
        """Search for similar documents.

        Args:
            query: Query string or embedding vector.
            k: Number of results to return.
            filters: Filters to apply to the search.
            **kwargs: Additional arguments.

        Returns:
            List of document chunks that are similar to the query.
        """
        if not self._initialized:
            await self.initialize()

        # Convert query to vector if needed
        if isinstance(query, str):
            # Use the provided embedding function or the default one
            embedding_func = kwargs.get("embedding_function")
            if not callable(embedding_func):
                raise ValueError(
                    "An embedding function must be provided when querying with text"
                )
            query_vector = embedding_func(query)
        else:
            query_vector = query

        # Build filter expression
        filter_expr = self._build_filter_expression(filters)

        # Search parameters
        search_params = {
            **self.config.search_params,
            **kwargs.get("search_params", {}),
        }

        # Execute search
        try:
            results = self._collection.search(
                data=[query_vector],
                anns_field=VECTOR_FIELD,
                param=search_params,
                limit=k,
                expr=filter_expr,
                output_fields=[
                    DOC_ID_FIELD,
                    CHUNK_ID_FIELD,
                    CONTENT_FIELD,
                    CONTENT_TYPE_FIELD,
                    METADATA_FIELD,
                ],
            )

            # Convert results to DocumentChunk objects
            chunks = []
            for hits in results:
                for hit in hits:
                    try:
                        metadata = hit.entity.get(METADATA_FIELD, {})
                        if isinstance(metadata, str):
                            import json
                            metadata = json.loads(metadata)

                        chunk = DocumentChunk(
                            id=hit.entity.get(CHUNK_ID_FIELD),
                            document_id=hit.entity.get(DOC_ID_FIELD),
                            content=hit.entity.get(CONTENT_FIELD, ""),
                            metadata=DocumentMetadata(
                                source=metadata.get("source"),
                                content_type=metadata.get("content_type"),
                                **{
                                    k: v
                                    for k, v in metadata.items()
                                    if k not in ["source", "content_type"]
                                },
                            ),
                            chunk_index=metadata.get("chunk_index", 0),
                            chunk_count=metadata.get("chunk_count", 1),
                            vector_embedding=hit.entity.get(VECTOR_FIELD),
                        )
                        chunks.append(chunk)
                    except Exception as e:
                        logger.error(f"Error processing search result: {e}")
                        continue

            return chunks

        except Exception as e:
            logger.error(f"Error searching in Milvus: {e}")
            raise

    def _build_filter_expression(self, filters: Optional[Dict[str, Any]] = None) -> str:
        """Build a filter expression for Milvus search.

        Args:
            filters: Dictionary of filters to apply.

        Returns:
            Filter expression string.
        """
        if not filters:
            return ""

        conditions = []
        for key, value in filters.items():
            if key == "content_type":
                if isinstance(value, list):
                    types = [f"'{v}'" for v in value]
                    conditions.append(f"{CONTENT_TYPE_FIELD} in [{', '.join(types)}]")
                else:
                    conditions.append(f"{CONTENT_TYPE_FIELD} == '{value}'")
            elif key == "source":
                conditions.append(f"{METADATA_FIELD}['source'] == '{value}'")
            elif key == "doc_id":
                conditions.append(f"{DOC_ID_FIELD} == '{value}'")
            else:
                logger.warning(f"Unsupported filter field: {key}")

        return " and ".join(conditions) if conditions else ""

    async def get_document(self, doc_id: str, **kwargs) -> Optional[Document]:
        """Get a document by ID.

        Args:
            doc_id: Document ID.
            **kwargs: Additional arguments.

        Returns:
            Document if found, None otherwise.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Query chunks for this document
            results = self._collection.query(
                expr=f"{DOC_ID_FIELD} == '{doc_id}'",
                output_fields=[
                    DOC_ID_FIELD,
                    CHUNK_ID_FIELD,
                    CONTENT_FIELD,
                    CONTENT_TYPE_FIELD,
                    METADATA_FIELD,
                ],
            )

            if not results:
                return None

            # Group chunks by document
            chunks = []
            doc_metadata = None

            for item in results:
                metadata = item.get(METADATA_FIELD, {})
                if isinstance(metadata, str):
                    import json
                    metadata = json.loads(metadata)

                if doc_metadata is None:
                    doc_metadata = DocumentMetadata(
                        source=metadata.get("source"),
                        content_type=metadata.get("content_type"),
                        **{
                            k: v
                            for k, v in metadata.items()
                            if k not in ["source", "content_type"]
                        },
                    )

                chunk = DocumentChunk(
                    id=item.get(CHUNK_ID_FIELD),
                    document_id=doc_id,
                    content=item.get(CONTENT_FIELD, ""),
                    metadata=doc_metadata,
                    chunk_index=metadata.get("chunk_index", 0),
                    chunk_count=metadata.get("chunk_count", 1),
                )
                chunks.append(chunk)

            # Sort chunks by index
            chunks.sort(key=lambda x: x.chunk_index)

            # Reconstruct document content
            content = "\n\n".join(chunk.content for chunk in chunks)

            return Document(
                id=doc_id,
                content=content,
                metadata=doc_metadata,
                chunks=chunks,
            )

        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None

    async def delete_document(self, doc_id: str, **kwargs) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: Document ID.
            **kwargs: Additional arguments.

        Returns:
            True if the document was deleted, False otherwise.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Delete all chunks for this document
            result = self._collection.delete(f"{DOC_ID_FIELD} == '{doc_id}'")
            logger.info(f"Deleted {result.delete_count} chunks for document {doc_id}")
            return result.delete_count > 0
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    async def get_stats(self, **kwargs) -> Dict[str, Any]:
        """Get statistics about the vector store.

        Args:
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing statistics.
        """
        if not self._initialized:
            await self.initialize()

        try:
            stats = {
                "collection": self.config.collection_name,
                "documents": 0,
                "chunks": 0,
                "dimension": self.config.dimension,
                "index_type": self.config.index_params.get("index_type", ""),
                "metric_type": self.config.index_params.get("metric_type", ""),
            }

            # Get collection statistics
            if self._collection:
                stats["chunks"] = self._collection.num_entities

                # Count unique document IDs
                try:
                    result = self._collection.query(
                        expr="",
                        output_fields=[f"count(distinct {DOC_ID_FIELD}) as doc_count"],
                    )
                    if result and "doc_count" in result[0]:
                        stats["documents"] = result[0]["doc_count"]
                except Exception as e:
                    logger.warning(f"Could not get document count: {e}")

            return stats

        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"error": str(e)}

    async def clear(self, **kwargs) -> None:
        """Clear the vector store.

        Args:
            **kwargs: Additional arguments.
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self._collection:
                # Delete all data from the collection
                self._collection.drop()
                logger.info(f"Dropped collection '{self.config.collection_name}'")
                # Recreate the collection
                self._create_collection()
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise

    async def close(self) -> None:
        """Close the vector store and release resources."""
        if self._collection:
            try:
                self._collection.release()
                connections.disconnect(self._connection_alias)
                logger.info("Disconnected from Milvus")
            except Exception as e:
                logger.error(f"Error closing Milvus connection: {e}")
            finally:
                self._collection = None
                self._initialized = False


# Register the Milvus vector store with the factory
VectorStoreFactory.register(VectorStoreType.MILVUS, MilvusVectorStore)
