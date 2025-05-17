"""HoarderMCP: A web content crawling and vector storage MCP server."""

__version__ = "0.1.0"
__author__ = "Your Name <your.email@example.com>"
__license__ = "MIT"

# Import key components for easier access
from .api import app
from .core import IngestService, VectorStoreService, ChunkingConfig, EmbeddingConfig
from .models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    IngestRequest,
    IngestSource,
    VectorSearchRequest,
    VectorSearchResult,
)

__all__ = [
    "app",
    "IngestService",
    "VectorStoreService",
    "ChunkingConfig",
    "EmbeddingConfig",
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "DocumentType",
    "IngestRequest",
    "IngestSource",
    "VectorSearchRequest",
    "VectorSearchResult",
]