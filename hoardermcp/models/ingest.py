"""Models for document ingestion."""
from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, Field, HttpUrl, validator


class IngestStatus(str, Enum):
    """Status of an ingestion job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestSource(BaseModel):
    """Source of content to ingest."""
    url: Optional[HttpUrl] = Field(
        None,
        description="URL to crawl (for web content)"
    )
    file_path: Optional[str] = Field(
        None,
        description="Path to a local file to ingest"
    )
    content: Optional[str] = Field(
        None,
        description="Direct content to ingest"
    )
    content_type: Optional[str] = Field(
        None,
        description="MIME type of the content"
    )

    @validator('*', pre=True)
    def check_exclusive_fields(cls, v, values, **kwargs):
        """Ensure only one of url, file_path, or content is provided."""
        field = kwargs['field'].name
        if field in ('url', 'file_path', 'content') and v is not None:
            others = {'url', 'file_path', 'content'} - {field}
            if any(values.get(f) is not None for f in others):
                raise ValueError(
                    f"Only one of {', '.join(others)} or {field} can be provided"
                )
        return v


class IngestRequest(BaseModel):
    """Request to ingest content."""
    sources: List[IngestSource] = Field(
        ...,
        description="Sources to ingest"
    )
    metadata: Optional[dict] = Field(
        None,
        description="Additional metadata to include with all documents"
    )
    chunk_size: int = Field(
        1000,
        description="Maximum size of each chunk in characters",
        ge=100,
        le=2000
    )
    chunk_overlap: int = Field(
        200,
        description="Number of characters to overlap between chunks",
        ge=0,
        le=500
    )
    store_original: bool = Field(
        True,
        description="Whether to store the original content"
    )
    vector_store: str = Field(
        "milvus",
        description="Name of the vector store to use"
    )

    class Config:
        schema_extra = {
            "example": {
                "sources": [
                    {
                        "url": "https://example.com/docs",
                        "content_type": "text/markdown"
                    }
                ],
                "metadata": {
                    "project": "example",
                    "version": "1.0.0"
                },
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "store_original": True,
                "vector_store": "milvus"
            }
        }
