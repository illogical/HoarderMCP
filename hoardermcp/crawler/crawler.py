"""Web crawler implementation using Crawl4AI with sitemap support."""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any, Tuple

from crawl4ai import WebCrawler, CrawlerConfig, Request
from crawl4ai.crawler_strategy import StaticCrawlerStrategy
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    ExtractedData,
)
from pydantic import HttpUrl, ValidationError

from ..models.document import Document, DocumentMetadata, DocumentType
from .extractors import ContentExtractor, get_extractor
from .sitemap_crawler import SitemapCrawler, extract_sitemap_urls, crawl_sitemap_to_documents

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
    status_code: int
    metadata: dict
    links: List[str] = field(default_factory=list)
    error: Optional[str] = None
    source_type: str = "direct"  # 'direct', 'sitemap', 'sitemap_index'
    sitemap_metadata: Optional[Dict[str, Any]] = None

    @property
    def success(self) -> bool:
        """Whether the crawl was successful."""
        return self.status_code == 200 and self.error is None

    def to_document(self, **metadata) -> Document:
        """Convert to a Document object."""
        # Merge metadata from sitemap if available
        additional_metadata = {}
        if self.sitemap_metadata:
            additional_metadata.update({
                "sitemap_lastmod": self.sitemap_metadata.get("lastmod"),
                "sitemap_changefreq": self.sitemap_metadata.get("changefreq"),
                "sitemap_priority": self.sitemap_metadata.get("priority"),
            })
            
            # Add images and videos from sitemap if available
            if "images" in self.sitemap_metadata and self.sitemap_metadata["images"]:
                additional_metadata["images"] = self.sitemap_metadata["images"]
                
            if "videos" in self.sitemap_metadata and self.sitemap_metadata["videos"]:
                additional_metadata["videos"] = self.sitemap_metadata["videos"]

        # Create document metadata
        doc_metadata = DocumentMetadata(
            source=self.url,
            content_type=self._infer_document_type(),
            last_crawled=datetime.utcnow().isoformat(),
            status_code=self.status_code,
            **{**additional_metadata, **metadata},
        )

        return Document(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, self.url)),
            content=self.content,
            metadata=doc_metadata,
        )

    def _infer_document_type(self) -> DocumentType:
        """Infer the document type from the content type or URL."""
        content_type = (self.content_type or "").lower()
        
        if "text/html" in content_type or not content_type and self.url.startswith(("http://", "https://")):
            return DocumentType.WEB_PAGE
        elif "application/pdf" in content_type or self.url.lower().endswith(".pdf"):
            return DocumentType.PDF
        elif self.url.lower().endswith(".py"):
            return DocumentType.PYTHON
        elif self.url.lower().endswith(".cs"):
            return DocumentType.CSHARP
        elif self.url.lower().endswith(".md") or "markdown" in content_type:
            return DocumentType.MARKDOWN
        elif "json" in content_type or self.url.lower().endswith(".json"):
            return DocumentType.JSON
        elif "xml" in content_type or self.url.lower().endswith(".xml"):
            return DocumentType.XML
        else:
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
        self._sitemap_crawler: Optional[SitemapCrawler] = None
        self._session = None  # Will be initialized when needed
        
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None
            
    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _crawl_single(
        self, 
        url: str, 
        options: CrawlOptions,
        sitemap_metadata: Optional[Dict[str, Any]] = None
    ) -> CrawlResult:
        """Crawl a single URL.

        Args:
            url: URL to crawl.
            options: Crawling options.
            sitemap_metadata: Optional metadata from sitemap.

        Returns:
            Crawl result.
        """
        try:
            # Check if this is a sitemap URL that we should parse
            if self._is_sitemap_url(url):
                return await self._crawl_sitemap(url, options)
                
            # Use Crawl4AI to crawl the URL
            crawler = WebCrawler()
            result = crawler.crawl(
                url=url,
                config=CrawlerConfig(
                    max_pages=options.max_pages,
                    max_depth=options.max_depth,
                    include_links=options.include_links,
                    include_images=options.include_images,
                    include_tables=options.include_tables,
                    include_iframes=options.include_iframes,
                    include_scripts=options.include_scripts,
                    include_styles=options.include_styles,
                    include_comments=options.include_comments,
                    include_meta=options.include_meta,
                    include_structured_data=options.include_structured_data,
                    include_embeds=options.include_embeds,
                    include_media=options.include_media,
                    include_forms=options.include_forms,
                    include_buttons=options.include_buttons,
                    include_inputs=options.include_inputs,
                    include_links_to=options.include_links_to,
                    exclude_links=options.exclude_links,
                ),
                strategy=StaticCrawlerStrategy(),
                extract_strategy=LLMExtractionStrategy(
                    model="gpt-4",  # Use the appropriate model
                    temperature=0.0,
                    max_tokens=2000,
                ),
            )

            # Convert to our result format
            crawl_result = CrawlResult(
                url=url,
                content=result.content,
                content_type=result.content_type or "text/html",
                status_code=result.status_code or 200,
                metadata={
                    "title": getattr(result, 'title', ''),
                    "description": getattr(result, 'description', ''),
                    "language": getattr(result, 'language', ''),
                    "encoding": getattr(result, 'encoding', 'utf-8'),
                    "links": [link.get("href", "") for link in getattr(result, 'links', [])],
                },
                links=[link.get("href", "") for link in getattr(result, 'links', [])],
                source_type="sitemap" if sitemap_metadata else "direct",
                sitemap_metadata=sitemap_metadata,
            )
            
            return crawl_result

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}", exc_info=True)
            return CrawlResult(
                url=url,
                content="",
                content_type="",
                status_code=0,
                metadata={"error": str(e)},
                error=str(e),
                source_type="sitemap" if sitemap_metadata else "direct",
                sitemap_metadata=sitemap_metadata,
            )
            
    async def _crawl_sitemap(self, sitemap_url: str, options: CrawlOptions) -> CrawlResult:
        """Crawl a sitemap and return a result with the sitemap contents.
        
        Args:
            sitemap_url: URL of the sitemap
            options: Crawling options
            
        Returns:
            CrawlResult with sitemap contents
        """
        try:
            # Use our sitemap crawler to get all URLs
            sitemap_crawler = SitemapCrawler(sitemap_url, session=self._get_session())
            sitemap_urls = await sitemap_crawler.crawl()
            
            # Create a summary of the sitemap
            content = f"Sitemap: {sitemap_url}\n"
            content += f"Total URLs: {len(sitemap_urls)}\n\n"
            
            # Add URLs with their metadata
            for i, url in enumerate(sitemap_urls[:100], 1):  # Limit to first 100 URLs
                content += f"{i}. {url.loc}"
                if url.lastmod:
                    content += f" (lastmod: {url.lastmod})"
                if url.priority is not None:
                    content += f" [priority: {url.priority}]"
                content += "\n"
                
            if len(sitemap_urls) > 100:
                content += f"\n... and {len(sitemap_urls) - 100} more URLs"
                
            return CrawlResult(
                url=sitemap_url,
                content=content,
                content_type="text/plain",
                status_code=200,
                metadata={
                    "sitemap": True,
                    "url_count": len(sitemap_urls),
                    "urls": [url.loc for url in sitemap_urls],
                },
                source_type="sitemap_index",
            )
            
        except Exception as e:
            logger.error(f"Error processing sitemap {sitemap_url}: {e}", exc_info=True)
            return CrawlResult(
                url=sitemap_url,
                content=f"Error processing sitemap: {e}",
                content_type="text/plain",
                status_code=500,
                metadata={"error": str(e)},
                error=str(e),
                source_type="sitemap_index",
            )

    async def crawl(
        self, urls: Union[str, List[str]], **options
    ) -> List[CrawlResult]:
        """Crawl one or more URLs.

        Args:
            urls: URL or list of URLs to crawl. Can also be a sitemap URL.
            **options: Override default crawl options.

        Returns:
            List of crawl results.
        """
        if isinstance(urls, str):
            urls = [urls]

        # Update options with any overrides
        crawl_options = self.options
        if options:
            crawl_options = self.options.__class__(**{**self.options.__dict__, **options})
            
        # Check if any URL is a sitemap
        sitemap_urls = [url for url in urls if self._is_sitemap_url(url)]
        if sitemap_urls:
            return await self.crawl_sitemaps(sitemap_urls, **options)
            
        # Regular URL crawling
        results = []
        for url in urls:
            if url in self._seen_urls:
                continue

            self._seen_urls.add(url)
            result = await self._crawl_single(url, crawl_options)
            results.append(result)
            
            if crawl_options.output_dir:
                self._save_result(result, Path(crawl_options.output_dir))

        return results

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

    async def crawl_sitemap(self, url: str) -> List[Document]:
        """Crawl a sitemap and return the documents it contains.
        
        This method will:
        1. Discover all sitemaps for the given URL (including from robots.txt)
        2. Parse all sitemaps (including sitemap indexes)
        3. Convert sitemap entries to Document objects
        
        Args:
            url: Base URL of the website or direct sitemap URL
            
        Returns:
            List of Document objects with metadata from sitemap
        """
        from .sitemap import SitemapCrawler
        
        # Use an existing session if available
        session = getattr(self, '_session', None)
        
        async with SitemapCrawler(url, session=session) as sitemap_crawler:
            # First discover all sitemaps
            logger.info(f"Discovering sitemaps for {url}")
            await sitemap_crawler.discover_sitemaps()
            
            if not sitemap_crawler.discovered_sitemaps:
                logger.warning(f"No sitemaps found for {url}")
                return []
                
            logger.info(f"Found {len(sitemap_crawler.discovered_sitemaps)} sitemaps")
            
            # Parse all sitemaps and get documents
            documents = await sitemap_crawler.to_documents()
            logger.info(f"Extracted {len(documents)} URLs from sitemaps")
            
            # Cache the session for potential reuse
            if not hasattr(self, '_session'):
                self._session = sitemap_crawler.session
                self._external_session = False
            
            return documents

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
