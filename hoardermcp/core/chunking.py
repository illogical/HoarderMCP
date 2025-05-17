"""Advanced chunking strategies for different content types."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from lxml import etree
from markdown import markdown
from pydantic import BaseModel, Field

from ..models.document import Document, DocumentChunk, DocumentType


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""
    SEMANTIC = "semantic"
    CODE = "code"
    FIXED = "fixed"


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.SEMANTIC,
        description="Chunking strategy to use",
    )
    chunk_size: int = Field(
        default=1000,
        description="Maximum size of each chunk in characters",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=200,
        description="Number of characters to overlap between chunks",
        ge=0,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens per chunk (if token counting is used)",
        gt=0,
    )
    language: Optional[Language] = Field(
        default=None,
        description="Programming language for code chunking",
    )
    separators: Optional[List[str]] = Field(
        default=None,
        description="Custom separators for text splitting",
    )

    class Config:
        """Pydantic config."""

        extra = "forbid"
        use_enum_values = True


class ChunkingError(Exception):
    """Base class for chunking errors."""

    pass


class ChunkingFactory:
    """Factory for creating chunkers based on document type and strategy."""

    @classmethod
    def get_chunker(
        cls, doc_type: Union[str, DocumentType], config: ChunkingConfig
    ) -> "BaseChunker":
        """Get a chunker for the given document type and strategy.

        Args:
            doc_type: Type of the document.
            config: Chunking configuration.

        Returns:
            A chunker instance.
        """
        doc_type = DocumentType(doc_type)
        strategy = ChunkingStrategy(config.strategy)

        if doc_type in (DocumentType.PYTHON, DocumentType.CSHARP):
            return CodeChunker(config)
        elif doc_type == DocumentType.MARKDOWN:
            if strategy == ChunkingStrategy.SEMANTIC:
                return MarkdownSemanticChunker(config)
            return MarkdownChunker(config)
        else:
            return TextChunker(config)


class BaseChunker:
    """Base class for chunkers."""

    def __init__(self, config: ChunkingConfig):
        """Initialize the chunker.

        Args:
            config: Chunking configuration.
        """
        self.config = config
        self._splitter = self._create_splitter()

    def _create_splitter(self) -> TextSplitter:
        """Create a text splitter based on the configuration.

        Returns:
            A text splitter instance.
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators or ["\n\n", "\n", " ", ""],
            keep_separator=True,
        )

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a document into chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of document chunks.
        """
        texts = self._splitter.split_text(document.content)
        return self._create_chunks(document, texts)

    def _create_chunks(
        self, document: Document, texts: List[str]
    ) -> List[DocumentChunk]:
        """Create document chunks from split texts.

        Args:
            document: Original document.
            texts: List of split text chunks.

        Returns:
            List of document chunks.
        """
        return [
            DocumentChunk(
                id=f"{document.id}_chunk_{i}",
                document_id=document.id,
                content=text,
                metadata=document.metadata.copy(),
                chunk_index=i,
                chunk_count=len(texts),
            )
            for i, text in enumerate(texts)
        ]


class TextChunker(BaseChunker):
    """Chunker for plain text documents."""

    pass


class CodeChunker(BaseChunker):
    """Chunker for code documents."""

    def _create_splitter(self) -> TextSplitter:
        """Create a code-aware text splitter.

        Returns:
            A text splitter instance configured for code.
        """
        language = self.config.language or self._detect_language()
        
        # Use language-specific splitters when available
        if language == Language.PYTHON:
            return RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        
        # Fall back to generic code splitter
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", " ", ""],
            keep_separator=True,
        )

    def _detect_language(self) -> Language:
        """Detect the programming language from the configuration or document type.

        Returns:
            Detected language.
        """
        if self.config.language:
            return self.config.language
        
        # Default to Python if no language is specified
        return Language.PYTHON


class MarkdownChunker(BaseChunker):
    """Chunker for Markdown documents with basic splitting."""

    def _create_splitter(self) -> TextSplitter:
        """Create a markdown-aware text splitter.

        Returns:
            A text splitter instance configured for markdown.
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""],
            keep_separator=True,
        )


class MarkdownSemanticChunker(MarkdownChunker):
    """Chunker for Markdown documents with semantic splitting."""

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a markdown document into semantic chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of document chunks.
        """
        # First, split by sections using headers
        sections = self._split_into_sections(document.content)
        
        # Then, split each section into chunks
        all_chunks = []
        for section_text, section_metadata in sections:
            # Create a temporary document for the section
            section_doc = document.copy()
            section_doc.content = section_text
            section_doc.metadata = section_doc.metadata.copy()
            section_doc.metadata.update(section_metadata)
            
            # Split the section into chunks
            chunks = super().chunk_document(section_doc)
            all_chunks.extend(chunks)
        
        return all_chunks

    def _split_into_sections(self, content: str) -> List[Tuple[str, dict]]:
        """Split markdown content into sections based on headers.

        Args:
            content: Markdown content.

        Returns:
            List of (section_text, section_metadata) tuples.
        """
        # Convert markdown to HTML and parse it
        html = markdown(content, extensions=['extra', 'tables'])
        root = etree.HTML(f"<div>{html}</div>")
        
        sections = []
        current_section = []
        current_metadata = {}
        
        # Process each element in the document
        for element in root.xpath('//div/*'):
            if element.tag.startswith('h') and element.tag[1:].isdigit():
                # Save previous section if it exists
                if current_section:
                    section_text = '\n'.join(current_section)
                    sections.append((section_text, current_metadata.copy()))
                    current_section = []
                
                # Start new section with header metadata
                header_level = int(element.tag[1])
                header_text = ''.join(element.itertext()).strip()
                current_metadata = {
                    'header_level': header_level,
                    'header_text': header_text,
                }
                
                # Add header to section text
                current_section.append(f"{'#' * header_level} {header_text}")
            else:
                # Add element text to current section
                if element.text and element.text.strip():
                    current_section.append(element.text.strip())
                
                # Add tail text if it exists
                if element.tail and element.tail.strip():
                    current_section.append(element.tail.strip())
        
        # Add the last section
        if current_section:
            section_text = '\n'.join(current_section)
            sections.append((section_text, current_metadata))
        
        return sections
