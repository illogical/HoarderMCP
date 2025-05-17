"""Base classes for vector stores."""
from __future__ import annotations

import abc
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic

from pydantic import BaseModel, Field, validator

from ..models.document import Document, DocumentChunk, DocumentType

T = TypeVar("T", bound="VectorStoreConfig")


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    MILVUS = "milvus"
    FAISS = "faiss"
    CHROMA = "chroma"
    SUPABASE = "supabase"
    MEMORY = "memory"


class VectorStoreConfig(BaseModel):
    """Base configuration for vector stores."""

    type: VectorStoreType = Field(..., description="Type of vector store")
    collection_name: str = Field(
        "hoardermcp_documents",
        description="Name of the collection/namespace to use",
    )
    dimension: int = Field(
        384,  # Default for all-MiniLM-L6-v2
        description="Dimensionality of the vector embeddings",
        gt=0,
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the vector store",
    )
    content_types: List[DocumentType] = Field(
        default_factory=lambda: list(DocumentType),
        description="Content types to store in this vector store",
    )

    class Config:
        """Pydantic config."""

        extra = "forbid"
        use_enum_values = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VectorStoreConfig:
        """Create a config from a dictionary.

        Args:
            data: Dictionary containing the configuration.

        Returns:
            VectorStoreConfig instance.
        """
        if "type" not in data:
            raise ValueError("Missing required field 'type'")

        # Map type string to config class
        type_map = {
            VectorStoreType.MILVUS: MilvusConfig,
            # Add other config types here
        }

        config_cls = type_map.get(VectorStoreType(data["type"]), cls)
        return config_cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> VectorStoreConfig:
        """Create a config from a JSON string.

        Args:
            json_str: JSON string containing the configuration.

        Returns:
            VectorStoreConfig instance.
        """
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary.

        Returns:
            Dictionary representation of the config.
        """
        return self.dict()

    def to_json(self) -> str:
        """Convert the config to a JSON string.

        Returns:
            JSON string representation of the config.
        """
        return self.json()


class VectorStore(abc.ABC, Generic[T]):
    """Abstract base class for vector stores."""

    def __init__(self, config: T):
        """Initialize the vector store.

        Args:
            config: Configuration for the vector store.
        """
        self.config = config
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the vector store is initialized.

        Returns:
            True if initialized, False otherwise.
        """
        return self._initialized

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store.

        This should be called before any other methods.
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    async def get_document(self, doc_id: str, **kwargs) -> Optional[Document]:
        """Get a document by ID.

        Args:
            doc_id: Document ID.
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            Document if found, None otherwise.
        """
        pass

    @abc.abstractmethod
    async def delete_document(self, doc_id: str, **kwargs) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: Document ID.
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            True if the document was deleted, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def get_stats(self, **kwargs) -> Dict[str, Any]:
        """Get statistics about the vector store.

        Args:
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            Dictionary containing statistics.
        """
        pass

    @abc.abstractmethod
    async def clear(self, **kwargs) -> None:
        """Clear the vector store.

        Args:
            **kwargs: Additional arguments to pass to the vector store.
        """
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the vector store and release resources."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class VectorStoreFactory:
    """Factory for creating vector stores."""

    _registry: Dict[VectorStoreType, Type[VectorStore]] = {}

    @classmethod
    def register(
        cls, store_type: VectorStoreType, store_cls: Type[VectorStore]
    ) -> None:
        """Register a vector store class.

        Args:
            store_type: Type of vector store.
            store_cls: Vector store class.
        """
        if not issubclass(store_cls, VectorStore):
            raise TypeError(f"{store_cls.__name__} is not a subclass of VectorStore")
        cls._registry[store_type] = store_cls

    @classmethod
    def create(
        cls, config: VectorStoreConfig, **kwargs
    ) -> VectorStore:
        """Create a vector store.

        Args:
            config: Vector store configuration.
            **kwargs: Additional arguments to pass to the vector store constructor.

        Returns:
            Vector store instance.

        Raises:
            ValueError: If the vector store type is not registered.
        """
        store_cls = cls._registry.get(config.type)
        if not store_cls:
            raise ValueError(f"No vector store registered for type '{config.type}'")
        return store_cls(config, **kwargs)


# Import MilvusConfig here to avoid circular imports
from .milvus_store import MilvusConfig  # noqa: E402
