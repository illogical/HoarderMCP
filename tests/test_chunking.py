"""Tests for document chunking strategies."""
from pathlib import Path
from typing import List

import pytest

from hoardermcp.core.chunking import (
    ChunkingConfig,
    ChunkingFactory,
    ChunkingStrategy,
    CodeChunker,
    MarkdownSemanticChunker,
    TextChunker,
)
from hoardermcp.models.document import Document, DocumentMetadata, DocumentType

# Sample test data
SAMPLE_MARKDOWN = """# Title

This is a sample markdown document.

## Section 1

This is section 1 content.

## Section 2

This is section 2 content with some `code`.
"""

SAMPLE_PYTHON = """def hello():
    """A simple function."""
    print("Hello, World!")


class TestChunkingExample:
    """Example test class for chunking functionality."""

    def test_example_method(self):
        """Example test method."""
        pass
"""

SAMPLE_TEXT = """This is a sample text document.

It has multiple paragraphs.

And some more text here."""


def test_markdown_semantic_chunking():
    """Test semantic chunking of markdown documents."""
    # Setup
    config = ChunkingConfig(
        strategy=ChunkingStrategy.SEMANTIC,
        chunk_size=100,
        chunk_overlap=20,
    )
    doc = Document(
        id="test",
        content=SAMPLE_MARKDOWN,
        metadata=DocumentMetadata(
            source="test.md",
            content_type=DocumentType.MARKDOWN,
        ),
    )
    
    # Execute
    chunker = MarkdownSemanticChunker(config)
    chunks = chunker.chunk_document(doc)

    # Verify chunks were created correctly
    assert len(chunks) > 1  # Should be split into multiple chunks
    assert all(chunk.content for chunk in chunks)
    # No headers in the middle of chunks
    assert all(
        "# " not in chunk.content[1:]
        for chunk in chunks
    )


def test_code_chunking():
    """Test code chunking with Python code."""
    # Setup
    config = ChunkingConfig(
        strategy=ChunkingStrategy.CODE,
        chunk_size=100,
        chunk_overlap=20,
        language="python",
    )
    doc = Document(
        id="test",
        content=SAMPLE_PYTHON,
        metadata=DocumentMetadata(
            source="test.py",
            content_type=DocumentType.PYTHON,
        ),
    )
    
    # Execute
    chunker = CodeChunker(config)
    chunks = chunker.chunk_document(doc)
    # Verify chunks contain code structure
    assert len(chunks) > 1
    assert all(
        "def " in chunk.content or "class " in chunk.content
        for chunk in chunks
    )


def test_text_chunking():
    """Test basic text chunking."""
    # Setup
    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED,
        chunk_size=50,
        chunk_overlap=10,
    )
    doc = Document(
        id="test",
        content=SAMPLE_TEXT,
        metadata=DocumentMetadata(
            source="test.txt",
            content_type=DocumentType.TEXT,
        ),
    )
    
    # Execute
    chunker = TextChunker(config)
    chunks = chunker.chunk_document(doc)
    
    # Verify chunk size constraints
    assert len(chunks) > 1
    # Respects chunk size
    assert all(10 < len(chunk.content) <= 50 for chunk in chunks)


def test_chunking_factory():
    """Test the chunking factory creates the right chunker types."""
    config = ChunkingConfig()
    
    # Test markdown
    markdown_chunker = ChunkingFactory.get_chunker("markdown", config)
    assert isinstance(markdown_chunker, MarkdownSemanticChunker)

    # Test python
    python_chunker = ChunkingFactory.get_chunker("python", config)
    assert isinstance(python_chunker, CodeChunker)

    # Test text
    text_chunker = ChunkingFactory.get_chunker("text", config)
    assert isinstance(text_chunker, TextChunker)


def test_chunking_with_metadata():
    """Test that chunk metadata is properly set."""
    # Setup
    config = ChunkingConfig(
        strategy=ChunkingStrategy.SEMANTIC,
        chunk_size=100,
        chunk_overlap=20,
    )
    doc = Document(
        id="test_doc",
        content="# Header\n\nContent",
        metadata=DocumentMetadata(
            source="test.md",
            content_type=DocumentType.MARKDOWN,
            title="Test Document",
        ),
    )
    
    # Execute
    chunker = MarkdownSemanticChunker(config)
    chunks = chunker.chunk_document(doc)
    
    # Verify metadata
    assert len(chunks) == 1
    assert chunks[0].document_id == "test_doc"
    assert chunks[0].metadata.title == "Test Document"
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_count == 1
