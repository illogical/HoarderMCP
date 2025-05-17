"""Vector store service for HoarderMCP."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from ..models.document import Document, DocumentChunk, DocumentType
from ..storage import (
    VectorStore,
    VectorStoreConfig,
    VectorStoreFactory,
    VectorStoreType,
)

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing vector stores."""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], VectorStoreConfig]] = None,
        store_type: Optional[Union[str, VectorStoreType]] = None,
    ):
        """Initialize the vector store service.

        Args:
            config: Vector store configuration. If None, default configuration will be used.
            store_type: Type of vector store to use. If None, will be determined from config.
        """
        self.config = self._parse_config(config, store_type)
        self._store: Optional[VectorStore] = None
        self._initialized = False

    @property
    def store(self) -> VectorStore:
        """Get the underlying vector store instance.

        Returns:
            Vector store instance.

        Raises:
            RuntimeError: If the vector store is not initialized.
        """
        if self._store is None:
            raise RuntimeError("Vector store is not initialized. Call initialize() first.")
        return self._store

    @property
    def is_initialized(self) -> bool:
        """Whether the vector store is initialized.

        Returns:
            True if initialized, False otherwise.
        """
        return self._initialized and self._store is not None

    @classmethod
    def from_config(
        cls, config: Union[Dict[str, Any], VectorStoreConfig]
    ) -> VectorStoreService:
        """Create a vector store service from a configuration.

        Args:
            config: Vector store configuration.

        Returns:
            VectorStoreService instance.
        """
        return cls(config=config)

    @classmethod
    def create(
        cls,
        store_type: Union[str, VectorStoreType],
        **kwargs,
    ) -> VectorStoreService:
        """Create a vector store service with the specified type and configuration.

        Args:
            store_type: Type of vector store to create.
            **kwargs: Additional configuration parameters.

        Returns:
            VectorStoreService instance.
        """
        return cls(config=kwargs, store_type=store_type)

    def _parse_config(
        self,
        config: Optional[Union[Dict[str, Any], VectorStoreConfig]] = None,
        store_type: Optional[Union[str, VectorStoreType]] = None,
    ) -> VectorStoreConfig:
        """Parse and validate the vector store configuration.

        Args:
            config: Vector store configuration.
            store_type: Type of vector store to use.

        Returns:
            VectorStoreConfig instance.
        """
        if config is None:
            config = {}

        if isinstance(config, VectorStoreConfig):
            if store_type is not None and config.type != store_type:
                raise ValueError(
                    f"Store type mismatch: {config.type} != {store_type}"
                )
            return config

        if isinstance(store_type, str):
            store_type = VectorStoreType(store_type.lower())
        elif store_type is None:
            store_type = VectorStoreType.MILVUS  # Default to Milvus

        config_dict = dict(config)
        config_dict["type"] = store_type

        return VectorStoreConfig.from_dict(config_dict)

    async def initialize(self) -> None:
        """Initialize the vector store.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self._initialized:
            return

        try:
            # Create vector store instance
            self._store = VectorStoreFactory.create(self.config)

            # Initialize the store
            await self._store.initialize()
            self._initialized = True

            logger.info(
                f"Initialized {self.config.type.value} vector store "
                f"with config: {self.config.dict()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self._initialized = False
            raise RuntimeError(f"Failed to initialize vector store: {e}") from e

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
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            List of document IDs that were added.
        """
        if not self._initialized:
            await self.initialize()

        return await self.store.add_documents(
            documents=documents, batch_size=batch_size, **kwargs
        )

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
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            List of chunk IDs that were added.
        """
        if not self._initialized:
            await self.initialize()

        return await self.store.add_chunks(chunks=chunks, batch_size=batch_size, **kwargs)

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
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            List of document chunks that are similar to the query.
        """
        if not self._initialized:
            await self.initialize()

        return await self.store.similarity_search(
            query=query, k=k, filters=filters, **kwargs
        )

    async def get_document(self, doc_id: str, **kwargs) -> Optional[Document]:
        """Get a document by ID.

        Args:
            doc_id: Document ID.
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            Document if found, None otherwise.
        """
        if not self._initialized:
            await self.initialize()

        return await self.store.get_document(doc_id=doc_id, **kwargs)

    async def delete_document(self, doc_id: str, **kwargs) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: Document ID.
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            True if the document was deleted, False otherwise.
        """
        if not self._initialized:
            await self.initialize()

        return await self.store.delete_document(doc_id=doc_id, **kwargs)

    async def get_stats(self, **kwargs) -> Dict[str, Any]:
        """Get statistics about the vector store.

        Args:
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            Dictionary containing statistics.
        """
        if not self._initialized:
            await self.initialize()

        return await self.store.get_stats(**kwargs)

    async def clear(self, **kwargs) -> None:
        """Clear the vector store.

        Args:
            **kwargs: Additional arguments to pass to the vector store.
        """
        if not self._initialized:
            await self.initialize()

        await self.store.clear(**kwargs)

    async def close(self) -> None:
        """Close the vector store and release resources."""
        if self._store is not None:
            await self.store.close()
            self._store = None
            self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
