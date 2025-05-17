"""Sitemap crawler for HoarderMCP.

This module provides functionality to crawl and parse sitemaps, including sitemap indexes,
and extract URLs with their metadata. It supports both XML sitemaps and sitemap indexes.
"""
from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from urllib.parse import urlparse, urljoin, urlunparse

import aiohttp
from pydantic import BaseModel, Field, HttpUrl

from ..models import Document, DocumentMetadata, DocumentType

logger = logging.getLogger(__name__)

# Common sitemap locations to check
COMMON_SITEMAP_PATHS = [
    '/sitemap.xml',
    '/sitemap_index.xml',
    '/sitemap/sitemap.xml',
    '/sitemap-index.xml',
    '/sitemap/sitemap_index.xml',
]

# XML namespaces for sitemaps
SITEMAP_NS = {
    'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9',
    'image': 'http://www.google.com/schemas/sitemap-image/1.1',
    'video': 'http://www.google.com/schemas/sitemap-video/1.1',
}

# Register namespaces for pretty printing
for prefix, uri in SITEMAP_NS.items():
    ET.register_namespace(prefix, uri)


@dataclass
class SitemapUrl:
    """Represents a URL entry in a sitemap with extended metadata."""
    
    loc: str
    """The URL of the page."""
    
    lastmod: Optional[str] = None
    """The date of last modification of the file (ISO 8601 format)."""
    
    changefreq: Optional[str] = None
    """How frequently the page is likely to change."""
    
    priority: Optional[float] = None
    """Priority of this URL relative to other URLs on the site (0.0 to 1.0)."""
    
    images: List[Dict[str, str]] = field(default_factory=list)
    """List of images associated with this URL."""
    
    videos: List[Dict[str, Any]] = field(default_factory=list)
    """List of videos associated with this URL."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the sitemap URL.
        """
        return {
            "loc": self.loc,
            "lastmod": self.lastmod,
            "changefreq": self.changefreq,
            "priority": self.priority,
            "images": self.images or [],
            "videos": self.videos or [],
        }
    
    @classmethod
    def from_xml_element(cls, element: ET.Element) -> Optional[SitemapUrl]:
        """Create a SitemapUrl from an XML element.
        
        Args:
            element: XML element from a sitemap.
            
        Returns:
            SitemapUrl if the element is valid, None otherwise.
        """
        # Extract URL
        loc_elem = element.find("sm:loc", SITEMAP_NS)
        if loc_elem is None or not loc_elem.text:
            return None
            
        entry = cls(loc=loc_elem.text.strip())
        
        # Extract last modification date
        lastmod_elem = element.find("sm:lastmod", SITEMAP_NS)
        if lastmod_elem is not None and lastmod_elem.text:
            entry.lastmod = lastmod_elem.text.strip()
        
        # Extract change frequency
        changefreq_elem = element.find("sm:changefreq", SITEMAP_NS)
        if changefreq_elem is not None and changefreq_elem.text:
            entry.changefreq = changefreq_elem.text.lower().strip()
        
        # Extract priority
        priority_elem = element.find("sm:priority", SITEMAP_NS)
        if priority_elem is not None and priority_elem.text:
            try:
                entry.priority = float(priority_elem.text)
            except (ValueError, TypeError):
                pass
        
        # Extract images
        for img_elem in element.findall(".//image:image", SITEMAP_NS):
            img_loc = img_elem.find("image:loc", SITEMAP_NS)
            if img_loc is not None and img_loc.text:
                img = {"loc": img_loc.text.strip()}
                
                # Add optional image attributes
                for attr in ["caption", "geo_location", "title", "license"]:
                    elem = img_elem.find(f"image:{attr}", SITEMAP_NS)
                    if elem is not None and elem.text:
                        img[attr] = elem.text.strip()
                        
                entry.images.append(img)
        
        # Extract videos
        for video_elem in element.findall(".//video:video", SITEMAP_NS):
            video = {}
            
            # Add video attributes
            for attr in ["thumbnail_loc", "title", "description", "content_loc", "player_loc"]:
                elem = video_elem.find(f"video:{attr}", SITEMAP_NS)
                if elem is not None and elem.text:
                    video[attr] = elem.text.strip()
            
            # Add duration if present
            duration_elem = video_elem.find("video:duration", SITEMAP_NS)
            if duration_elem is not None and duration_elem.text:
                try:
                    video["duration"] = int(duration_elem.text)
                except (ValueError, TypeError):
                    pass
                    
            if video:
                entry.videos.append(video)
        
        return entry


class SitemapCrawler:
    """Crawler for sitemap.xml files with support for sitemap indexes and nested sitemaps.
    
    This crawler can discover sitemaps from a website's robots.txt, common sitemap locations,
    or directly from a provided sitemap URL. It handles both standard sitemaps and sitemap indexes.
    """

    def __init__(
        self, 
        base_url: str, 
        session: Optional[aiohttp.ClientSession] = None,
        max_retries: int = 3,
        request_timeout: int = 30,
    ):
        """Initialize the sitemap crawler.

        Args:
            base_url: Base URL of the website or direct sitemap URL.
            session: Optional aiohttp ClientSession to reuse
            max_retries: Maximum number of retries for failed requests
            request_timeout: Timeout for HTTP requests in seconds
        """
        self.base_url = self._normalize_url(base_url)
        self.domain = urlparse(self.base_url).netloc
        self.session = session or aiohttp.ClientSession()
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.visited_urls: Set[str] = set()
        self.discovered_sitemaps: Set[str] = set()
        self.sitemap_urls: List[SitemapUrl] = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit.
        
        Note: Does not close the session if it was passed in.
        """
        if self.session and not getattr(self, '_external_session', False):
            await self.session.close()
            
    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL by removing fragments and query parameters."""
        parsed = urlparse(url)
        return urlunparse((
            parsed.scheme or "https",
            parsed.netloc,
            parsed.path.rstrip("/") or "/",
            "",  # params
            "",  # query
            "",  # fragment
        ))
        
    async def discover_sitemaps(self) -> List[str]:
        """Discover sitemap files for the given base URL.
        
        Returns:
            List of discovered sitemap URLs.
        """
        sitemap_urls = set()
        
        # Check common sitemap locations
        tasks = []
        for path in COMMON_SITEMAP_PATHS:
            sitemap_url = urljoin(self.base_url, path)
            tasks.append(self._check_sitemap_url(sitemap_url))
        
        # Also check robots.txt for sitemap references
        tasks.append(self._check_robots_txt())
        
        # Run all checks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, list):
                sitemap_urls.update(result)
        
        self.discovered_sitemaps = sitemap_urls
        return list(sitemap_urls)
    
    async def _check_robots_txt(self) -> List[str]:
        """Check robots.txt for sitemap references."""
        robots_url = urljoin(self.base_url, "/robots.txt")
        try:
            async with self.session.get(robots_url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    text = await response.text()
                    # Find all Sitemap directives
                    sitemap_urls = []
                    for line in text.splitlines():
                        line = line.strip()
                        if line.lower().startswith("sitemap:"):
                            sitemap_url = line[8:].strip()
                            sitemap_urls.append(sitemap_url)
                    return sitemap_urls
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"Failed to fetch robots.txt: {e}")
        return []
    
    async def _check_sitemap_url(self, url: str) -> List[str]:
        """Check if a URL points to a valid sitemap."""
        try:
            async with self.session.head(url, timeout=self.request_timeout) as response:
                if response.status == 200 and \
                   response.headers.get("Content-Type", "").startswith(("application/xml", "text/xml")):
                    return [url]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"Failed to check sitemap URL {url}: {e}")
        return []
    
    async def crawl(self) -> List[SitemapUrl]:
        """Crawl and parse all discovered sitemaps.
        
        Returns:
            List of SitemapUrl objects from all sitemaps.
        """
        # First discover sitemaps if not already done
        if not self.discovered_sitemaps:
            await self.discover_sitemaps()
        
        if not self.discovered_sitemaps:
            logger.warning(f"No sitemaps found for {self.base_url}")
            return []
        
        # Parse all sitemaps in parallel
        tasks = [self.parse_sitemap(url) for url in self.discovered_sitemaps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten and deduplicate results
        all_urls = []
        seen = set()
        
        for result in results:
            if isinstance(result, list):
                for url in result:
                    if url.loc not in seen:
                        seen.add(url.loc)
                        all_urls.append(url)
        
        self.sitemap_urls = all_urls
        return all_urls
    
    async def parse_sitemap(self, sitemap_url: str) -> List[SitemapUrl]:
        """Parse a sitemap URL and return the URLs it contains.
        
        Args:
            sitemap_url: URL of the sitemap to parse
            
        Returns:
            List of SitemapUrl objects from the sitemap
        """
        if sitemap_url in self.visited_urls:
            return []
            
        self.visited_urls.add(sitemap_url)
        logger.info(f"Parsing sitemap: {sitemap_url}")
        
        try:
            async with self.session.get(sitemap_url, timeout=self.request_timeout) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch sitemap {sitemap_url}: HTTP {response.status}")
                    return []
                
                content = await response.text()
                
                # Check if this is a sitemap index
                if "sitemapindex" in content.lower():
                    return await self._parse_sitemap_index(content, sitemap_url)
                else:
                    return await self._parse_standard_sitemap(content, sitemap_url)
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Error parsing sitemap {sitemap_url}: {e}")
            return []
    
    async def _parse_sitemap_index(self, content: str, parent_url: str) -> List[SitemapUrl]:
        """Parse a sitemap index and return all referenced sitemaps."""
        try:
            root = ET.fromstring(content)
            sitemap_urls = []
            
            # Find all sitemap entries
            for sitemap in root.findall(".//sm:sitemap", SITEMAP_NS):
                loc_elem = sitemap.find("sm:loc", SITEMAP_NS)
                if loc_elem is not None and loc_elem.text:
                    sitemap_url = self._normalize_url(loc_elem.text)
                    sitemap_urls.append(sitemap_url)
            
            # Process all discovered sitemaps in parallel
            tasks = [self.parse_sitemap(url) for url in sitemap_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results
            all_urls = []
            for result in results:
                if isinstance(result, list):
                    all_urls.extend(result)
                    
            return all_urls
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap index {parent_url}: {e}")
            return []
    
    async def _parse_standard_sitemap(self, content: str, sitemap_url: str) -> List[SitemapUrl]:
        """Parse a standard sitemap and return all URLs."""
        try:
            root = ET.fromstring(content)
            urls = []
            
            for url_elem in root.findall(".//sm:url", SITEMAP_NS):
                sitemap_url = SitemapUrl.from_xml_element(url_elem)
                if sitemap_url:
                    urls.append(sitemap_url)
            
            return urls
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap {sitemap_url}: {e}")
            return []
    
    async def to_documents(self) -> List[Document]:
        """Convert sitemap URLs to Document objects.
        
        Returns:
            List of Document objects representing the sitemap entries.
        """
        if not self.sitemap_urls:
            await self.crawl()
            
        documents = []
        for url in self.sitemap_urls:
            # Create metadata
            metadata = DocumentMetadata(
                source=url.loc,
                content_type=DocumentType.WEB_PAGE,
                last_modified=url.lastmod,
                change_frequency=url.changefreq,
                priority=url.priority,
            )
            
            # Add additional metadata
            additional_metadata = {}
            
            if url.images:
                additional_metadata["images"] = url.images
                
            if url.videos:
                additional_metadata["videos"] = url.videos
                
            if additional_metadata:
                metadata.additional_metadata = additional_metadata
            
            # Create document with empty content (to be filled by the crawler)
            document = Document(
                id=url.loc,  # Use URL as ID
                content="",  # Content will be filled by the crawler
                metadata=metadata,
            )
            
            documents.append(document)
        
        return documents

        self._sitemap_urls: Set[str] = set()
        self._crawled_urls: Set[str] = set()
        self._sitemaps: Dict[str, List[SitemapEntry]] = {}
        self._client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; HoarderMCP/1.0; +https://github.com/yourusername/hoardermcp)"
            },
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def discover_sitemaps(self, url: str) -> List[str]:
        """Discover sitemap files for a given URL.

        Args:
            url: Website URL.

        Returns:
            List of discovered sitemap URLs.
        """
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        self.base_url = self.base_url or base_url

        # Common sitemap locations
        common_paths = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemaps/sitemap.xml",
            "/sitemap/sitemap.xml",
            "/sitemap",
        ]

        # Check robots.txt for sitemap
        try:
            robots_url = f"{base_url}/robots.txt"
            response = await self._client.get(robots_url)
            if response.status_code == 200:
                for line in response.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line[8:].strip()
                        self._sitemap_urls.add(sitemap_url)
        except Exception as e:
            logger.debug(f"Error checking robots.txt: {e}")

        # Check common sitemap locations
        for path in common_paths:
            sitemap_url = f"{base_url}{path}"
            if sitemap_url not in self._sitemap_urls:
                self._sitemap_urls.add(sitemap_url)

        # Try to fetch and parse each sitemap
        valid_sitemaps = []
        for sitemap_url in list(self._sitemap_urls):
            try:
                if await self._is_valid_sitemap(sitemap_url):
                    valid_sitemaps.append(sitemap_url)
                    # If this is a sitemap index, add all referenced sitemaps
                    await self._process_sitemap_index(sitemap_url)
            except Exception as e:
                logger.warning(f"Error processing sitemap {sitemap_url}: {e}")

        return valid_sitemaps

    async def _is_valid_sitemap(self, url: str) -> bool:
        """Check if a URL points to a valid sitemap.

        Args:
            url: URL to check.

        Returns:
            True if the URL points to a valid sitemap, False otherwise.
        """
        try:
            response = await self._client.head(url, timeout=10.0)
            if response.status_code != 200:
                return False

            content_type = response.headers.get("content-type", "").lower()
            if "xml" not in content_type and "text/xml" not in content_type:
                return False

            # Check the content to be sure
            response = await self._client.get(url, timeout=10.0)
            content = response.text.strip()
            return (
                content.startswith("<?xml")
                and "<sitemap" in content
                or "<urlset" in content
            )
        except Exception:
            return False

    async def _process_sitemap_index(self, url: str) -> None:
        """Process a sitemap index file.

        Args:
            url: URL of the sitemap index.
        """
        try:
            response = await self._client.get(url, timeout=10.0)
            response.raise_for_status()

            root = ET.fromstring(response.text)
            ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            # Find all sitemap entries
            for sitemap_elem in root.findall(".//ns:sitemap/ns:loc", ns):
                if sitemap_elem.text:
                    sitemap_url = sitemap_elem.text.strip()
                    if sitemap_url not in self._sitemap_urls:
                        self._sitemap_urls.add(sitemap_url)
                        # Recursively process nested sitemaps
                        await self._process_sitemap_index(sitemap_url)

        except Exception as e:
            logger.warning(f"Error processing sitemap index {url}: {e}")

    async def parse_sitemap(self, url: str) -> List[SitemapEntry]:
        """Parse a sitemap and return its entries.

        Args:
            url: URL of the sitemap.

        Returns:
            List of sitemap entries.
        """
        if url in self._sitemaps:
            return self._sitemaps[url]

        try:
            response = await self._client.get(url, timeout=10.0)
            response.raise_for_status()

            # Check if this is a sitemap index
            if b"<sitemapindex" in response.content:
                await self._process_sitemap_index(url)
                return []

            # Parse regular sitemap
            root = ET.fromstring(response.text)
            ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            entries = []
            for url_elem in root.findall(".//ns:url", ns):
                entry = SitemapEntry.from_xml_element(url_elem)
                if entry:
                    entries.append(entry)

            self._sitemaps[url] = entries
            return entries

        except Exception as e:
            logger.error(f"Error parsing sitemap {url}: {e}")
            return []

    async def crawl_sitemap(
        self, sitemap_url: str, options: Optional[CrawlOptions] = None
    ) -> List[CrawlResult]:
        """Crawl all URLs in a sitemap.

        Args:
            sitemap_url: URL of the sitemap.
            options: Crawling options.

        Returns:
            List of crawl results.
        """
        options = options or CrawlOptions()
        crawler = Crawler(options)

        # Discover all sitemaps
        await self.discover_sitemaps(sitemap_url)


        # Parse all sitemaps and collect URLs
        all_entries: List[SitemapEntry] = []
        for url in self._sitemap_urls:
            entries = await self.parse_sitemap(url)
            all_entries.extend(entries)

        # Sort by priority (descending) and lastmod (newest first)
        all_entries.sort(
            key=lambda x: (
                -1 if x.priority is None else -x.priority,
                x.lastmod or datetime.min,
            ),
            reverse=True,
        )

        # Limit to max_pages
        if options.max_pages > 0:
            all_entries = all_entries[: options.max_pages]

        # Convert to URLs and crawl
        urls = [entry.loc for entry in all_entries if entry.loc]
        results = await crawler.crawl(urls, **options.__dict__)

        return results

    async def crawl_website(
        self, url: str, options: Optional[CrawlOptions] = None
    ) -> List[CrawlResult]:
        """Crawl a website by first discovering its sitemaps.

        Args:
            url: Website URL.
            options: Crawling options.

        Returns:
            List of crawl results.
        """
        options = options or CrawlOptions()

        # First try to discover and crawl sitemaps
        sitemap_results = await self.crawl_sitemap(url, options)

        # If no sitemaps found, just crawl the homepage
        if not sitemap_results:
            logger.info("No sitemaps found, crawling homepage only")
            crawler = Crawler(options)
            return await crawler.crawl(url)

        return sitemap_results
