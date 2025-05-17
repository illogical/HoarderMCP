"""Vector search models for HoarderMCP."""
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, conint, conlist


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    MILVUS = "milvus"
    FAISS = "faiss"
    CHROMA = "chroma"
    SUPABASE = "supabase"


class VectorSearchRequest(BaseModel):
    """Request for vector similarity search."""
    query: str = Field(..., description="Search query")
    vector: Optional[List[float]] = Field(
        None,
        description="Pre-computed query vector (if not provided, will be generated)"
    )
    k: conint(ge=1, le=100) = Field(
        5,
        description="Number of results to return"
    )
    filters: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
        None,
        description="Filters to apply to the search"
    )
    include_metadata: bool = Field(
        True,
        description="Whether to include metadata in the results"
    )
    include_vectors: bool = Field(
        False,
        description="Whether to include vectors in the results"
    )

    class Config:
        schema_extra = {
            "example": {
                "query": "How to use async in Python?",
                "k": 5,
                "filters": {
                    "content_type": "python"
                },
                "include_metadata": True,
                "include_vectors": False
            }
        }


class VectorSearchResult(BaseModel):
    """A single search result from vector search."""
    id: str = Field(..., description="Unique identifier for the result")
    document_id: str = Field(..., description="ID of the parent document")
    content: str = Field(..., description="Content of the result")
    score: float = Field(..., description="Similarity score")
    metadata: Optional[Dict] = Field(None, description="Result metadata")
    vector: Optional[List[float]] = Field(None, description="Vector embedding")


class VectorSearchResponse(BaseModel):
    """Response from vector search."""
    results: List[VectorSearchResult] = Field(
        ...,
        description="Search results"
    )
    total: int = Field(..., description="Total number of results found")
    request_id: Optional[str] = Field(
        None,
        description="ID for tracing the request"
    )


class VectorStoreInfo(BaseModel):
    """Information about a vector store."""
    type: VectorStoreType = Field(..., description="Type of vector store")
    dimensions: int = Field(..., description="Dimensionality of vectors")
    count: int = Field(..., description="Number of vectors in the store")
    is_connected: bool = Field(..., description="Whether the store is connected")
    config: Dict = Field(..., description="Store configuration")
