"""Document models for HoarderMCP."""
from enum import Enum
from datetime import datetime
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, HttpUrl


class DocumentType(str, Enum):
    """Type of document content."""
    MARKDOWN = "markdown"
    PYTHON = "python"
    CSHARP = "csharp"
    HTML = "html"
    TEXT = "text"


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: HttpUrl = Field(..., description="Source URL of the document")
    content_type: DocumentType = Field(..., description="Type of content")
    title: Optional[str] = Field(None, description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    language: Optional[str] = Field("en", description="Document language code")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the document was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the document was last updated")
    metadata: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class DocumentChunk(BaseModel):
    """A chunk of a document with metadata."""
    id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="ID of the parent document")
    content: str = Field(..., description="Chunk content")
    metadata: DocumentMetadata = Field(..., description="Chunk metadata")
    chunk_index: int = Field(..., description="Index of this chunk in the document")
    chunk_count: int = Field(..., description="Total number of chunks in the document")
    vector_embedding: Optional[List[float]] = Field(
        None, 
        description="Vector embedding for this chunk"
    )


class Document(BaseModel):
    """A document with content and metadata."""
    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    chunks: List[DocumentChunk] = Field(
        default_factory=list,
        description="Document chunks (if split)"
    )

    @property
    def source(self) -> HttpUrl:
        """Get the source URL of the document."""
        return self.metadata.source

    @property
    def content_type(self) -> DocumentType:
        """Get the content type of the document."""
        return self.metadata.content_type
