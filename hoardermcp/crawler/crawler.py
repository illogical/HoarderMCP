"""Web crawler implementation using Crawl4AI."""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from crawl4ai import WebCrawler, CrawlerConfig, Request
from crawl4ai.crawler_strategy import StaticCrawlerStrategy
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    ExtractedData,
)
from pydantic import HttpUrl

from ..models.document import Document, DocumentMetadata, DocumentType
from .extractors import ContentExtractor, get_extractor

logger = logging.getLogger(__name__)


@dataclass
class CrawlOptions:
    """Options for web crawling."""

    max_pages: int = 10
    "Maximum number of pages to crawl per domain"

    max_depth: int = 2
    "Maximum depth to crawl"

    include_links: bool = True
    "Whether to include links in the crawled content"

    include_images: bool = False
    "Whether to include images in the crawled content"

    include_tables: bool = True
    "Whether to include tables in the crawled content"

    include_iframes: bool = False
    "Whether to include iframes in the crawled content"

    include_scripts: bool = False
    "Whether to include scripts in the crawled content"

    include_styles: bool = False
    "Whether to include styles in the crawled content"

    include_comments: bool = False
    "Whether to include comments in the crawled content"

    include_meta: bool = True
    "Whether to include meta tags in the crawled content"

    include_structured_data: bool = True
    "Whether to include structured data in the crawled content"

    include_embeds: bool = False
    "Whether to include embedded content in the crawled content"

    include_media: bool = False
    "Whether to include media in the crawled content"

    include_forms: bool = False
    "Whether to include forms in the crawled content"

    include_buttons: bool = False
    "Whether to include buttons in the crawled content"

    include_inputs: bool = False
    "Whether to include input fields in the crawled content"

    include_links_to: List[str] = field(default_factory=list)
    "List of URL patterns to include"

    exclude_links: List[str] = field(default_factory=list)
    "List of URL patterns to exclude"

    output_dir: Optional[Union[str, Path]] = None
    "Directory to save crawled content (if None, content is not saved to disk)"

    def to_crawl4ai_config(self) -> Dict:
        """Convert to Crawl4AI configuration."""
        return {
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "include_links": self.include_links,
            "include_images": self.include_images,
            "include_tables": self.include_tables,
            "include_iframes": self.include_iframes,
            "include_scripts": self.include_scripts,
            "include_styles": self.include_styles,
            "include_comments": self.include_comments,
            "include_meta": self.include_meta,
            "include_structured_data": self.include_structured_data,
            "include_embeds": self.include_embeds,
            "include_media": self.include_media,
            "include_forms": self.include_forms,
            "include_buttons": self.include_buttons,
            "include_inputs": self.include_inputs,
            "include_links_to": self.include_links_to,
            "exclude_links": self.exclude_links,
        }


@dataclass
class CrawlResult:
    """Result of a crawl operation."""

    url: str
    "URL that was crawled"

    content: str
    "Extracted content"

    content_type: str
    "MIME type of the content"

    status_code: int
    "HTTP status code"

    metadata: dict
    "Additional metadata"

    links: List[str] = field(default_factory=list)
    "List of links found on the page"

    error: Optional[str] = None
    "Error message if the crawl failed"

    @property
    def success(self) -> bool:
        """Whether the crawl was successful."""
        return self.error is None and self.status_code == 200

    def to_document(self, **metadata) -> Document:
        """Convert to a Document object."""
        doc_metadata = DocumentMetadata(
            source=self.url,
            content_type=self._infer_document_type(),
            **metadata,
        )

        return Document(
            id=str(uuid.uuid4()),
            content=self.content,
            metadata=doc_metadata,
        )

    def _infer_document_type(self) -> DocumentType:
        """Infer the document type from the content type."""
        content_type = self.content_type.lower()
        if "markdown" in content_type:
            return DocumentType.MARKDOWN
        elif "python" in content_type or ".py" in self.url:
            return DocumentType.PYTHON
        elif "csharp" in content_type or ".cs" in self.url:
            return DocumentType.CSHARP
        elif "html" in content_type:
            return DocumentType.HTML
        return DocumentType.TEXT


class Crawler:
    """Web crawler for extracting content from URLs."""

    def __init__(self, options: Optional[CrawlOptions] = None):
        """Initialize the crawler.

        Args:
            options: Crawling options. If None, default options will be used.
        """
        self.options = options or CrawlOptions()
        self._seen_urls: Set[str] = set()
        self._results: List[CrawlResult] = []

    async def crawl(
        self, urls: Union[str, List[str]], **options
    ) -> List[CrawlResult]:
        """Crawl one or more URLs.

        Args:
            urls: URL or list of URLs to crawl.
            **options: Override default crawl options.

        Returns:
            List of crawl results.
        """
        if isinstance(urls, str):
            urls = [urls]

        # Merge options
        crawl_options = self.options
        if options:
            crawl_options = CrawlOptions(**{**self.options.__dict__, **options})

        # Create output directory if needed
        if crawl_options.output_dir:
            output_dir = Path(crawl_options.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Configure crawler
        config = CrawlerConfig(
            **crawl_options.to_crawl4ai_config(),
            verbose=True,
        )

        # Create crawler
        crawler = WebCrawler(
            strategy=StaticCrawlerStrategy(config=config),
            extract_strategy=LLMExtractionStrategy(),
        )

        # Process URLs
        results = []
        for url in urls:
            if url in self._seen_urls:
                logger.debug(f"Skipping already seen URL: {url}")
                continue

            self._seen_urls.add(url)
            logger.info(f"Crawling URL: {url}")

            try:
                # Execute the crawl
                response = await crawler.crawl_async(Request(url=url))

                # Extract content
                extractor = get_extractor(response.content_type)
                content = extractor.extract(response)

                # Create result
                result = CrawlResult(
                    url=url,
                    content=content,
                    content_type=response.content_type,
                    status_code=response.status_code,
                    metadata={
                        "title": response.metadata.get("title"),
                        "description": response.metadata.get("description"),
                        "language": response.metadata.get("language"),
                        "crawled_at": datetime.utcnow().isoformat(),
                        "headers": dict(response.headers or {}),
                    },
                    links=response.links or [],
                )

                # Save to disk if output directory is specified
                if crawl_options.output_dir:
                    self._save_result(result, output_dir)

                results.append(result)

            except Exception as e:
                logger.error(f"Error crawling {url}: {e}", exc_info=True)
                results.append(
                    CrawlResult(
                        url=url,
                        content="",
                        content_type="",
                        status_code=0,
                        metadata={"error": str(e)},
                        error=str(e),
                    )
                )

        self._results.extend(results)
        return results

    def _save_result(self, result: CrawlResult, output_dir: Path) -> None:
        """Save a crawl result to disk.

        Args:
            result: Crawl result to save.
            output_dir: Directory to save the result to.
        """
        try:
            # Create a safe filename from the URL
            url_path = result.url.strip("/").replace("://", "_")
            safe_filename = "".join(
                c if c.isalnum() or c in ".-_" else "_" for c in url_path
            )
            if not safe_filename:
                safe_filename = f"page_{len(self._results) + 1}"

            # Add appropriate extension
            ext = ".txt"
            if "html" in result.content_type:
                ext = ".html"
            elif "markdown" in result.content_type:
                ext = ".md"
            elif "python" in result.content_type:
                ext = ".py"
            elif "csharp" in result.content_type:
                ext = ".cs"

            output_path = output_dir / f"{safe_filename}{ext}"

            # Write content to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.content)

            logger.debug(f"Saved crawl result to {output_path}")

        except Exception as e:
            logger.error(f"Error saving crawl result: {e}", exc_info=True)

    async def crawl_sitemap(
        self, sitemap_url: str, **options
    ) -> List[CrawlResult]:
        """Crawl all URLs in a sitemap.

        Args:
            sitemap_url: URL of the sitemap.
            **options: Additional crawl options.

        Returns:
            List of crawl results.
        """
        # TODO: Implement sitemap crawling
        raise NotImplementedError("Sitemap crawling not yet implemented")

    def get_results(self) -> List[CrawlResult]:
        """Get all crawl results.

        Returns:
            List of all crawl results.
        """
        return self._results.copy()

    def clear_results(self) -> None:
        """Clear all crawl results."""
        self._results.clear()
        self._seen_urls.clear()
