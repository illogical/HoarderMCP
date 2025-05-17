"""Content extractors for different document types."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, Union

from crawl4ai import ExtractedData

from ..models.document import DocumentType

logger = logging.getLogger(__name__)


class ContentExtractor(ABC):
    """Base class for content extractors."""

    content_type: str
    "MIME type of the content this extractor handles"

    document_type: DocumentType
    "Type of document this extractor handles"

    @classmethod
    @abstractmethod
    def extract(cls, data: Union[str, ExtractedData]) -> str:
        """Extract content from the given data.

        Args:
            data: Raw data to extract content from.

        Returns:
            Extracted content as a string.
        """
        pass


class MarkdownExtractor(ContentExtractor):
    """Extractor for Markdown content."""

    content_type = "text/markdown"
    document_type = DocumentType.MARKDOWN

    @classmethod
    def extract(cls, data: Union[str, ExtractedData]) -> str:
        if isinstance(data, str):
            return data
        return data.markdown or data.text or ""


class PythonExtractor(ContentExtractor):
    """Extractor for Python code."""

    content_type = "text/x-python"
    document_type = DocumentType.PYTHON

    @classmethod
    def extract(cls, data: Union[str, ExtractedData]) -> str:
        if isinstance(data, str):
            return data
        return data.text or data.markdown or ""


class CSharpExtractor(ContentExtractor):
    """Extractor for C# code."""

    content_type = "text/x-csharp"
    document_type = DocumentType.CSHARP

    @classmethod
    def extract(cls, data: Union[str, ExtractedData]) -> str:
        if isinstance(data, str):
            return data
        return data.text or data.markdown or ""


class TextExtractor(ContentExtractor):
    """Extractor for plain text content."""

    content_type = "text/plain"
    document_type = DocumentType.TEXT

    @classmethod
    def extract(cls, data: Union[str, ExtractedData]) -> str:
        if isinstance(data, str):
            return data
        return data.text or ""


class HTMLExtractor(ContentExtractor):
    """Extractor for HTML content."""

    content_type = "text/html"
    document_type = DocumentType.HTML

    @classmethod
    def extract(cls, data: Union[str, ExtractedData]) -> str:
        if isinstance(data, str):
            return data
        return data.markdown or data.text or ""


# Registry of available extractors
_EXTRACTORS: Dict[str, Type[ContentExtractor]] = {
    extractor.content_type: extractor
    for extractor in [
        MarkdownExtractor,
        PythonExtractor,
        CSharpExtractor,
        TextExtractor,
        HTMLExtractor,
    ]
}

# Map file extensions to content types
_EXTENSION_MAP = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".py": "text/x-python",
    ".cs": "text/x-csharp",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
}


def get_extractor(content_type: Optional[str] = None) -> Type[ContentExtractor]:
    """Get the appropriate extractor for the given content type.

    Args:
        content_type: MIME type of the content. If None, returns the TextExtractor.

    Returns:
        Content extractor class.
    """
    if not content_type:
        return TextExtractor

    # Normalize content type (remove parameters like charset)
    normalized_type = content_type.split(";")[0].strip().lower()

    # Try exact match first
    if normalized_type in _EXTRACTORS:
        return _EXTRACTORS[normalized_type]

    # Try partial match (e.g., "application/x-python" for "text/x-python")
    for key, extractor in _EXTRACTORS.items():
        if key in normalized_type or normalized_type in key:
            return extractor

    # Check file extensions
    if "." in normalized_type:
        ext = "." + normalized_type.split(".")[-1].lower()
        if ext in _EXTENSION_MAP:
            return _EXTRACTORS[_EXTENSION_MAP[ext]]

    # Default to text extractor
    return TextExtractor


def register_extractor(extractor: Type[ContentExtractor]) -> None:
    """Register a custom content extractor.

    Args:
        extractor: Extractor class to register.
    """
    if not issubclass(extractor, ContentExtractor):
        raise TypeError(
            f"Extractor must be a subclass of ContentExtractor, got {type(extractor)}"
        )
    _EXTRACTORS[extractor.content_type] = extractor
    logger.info(f"Registered extractor for {extractor.content_type}")
