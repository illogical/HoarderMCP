"""Sitemap crawler for HoarderMCP."""
from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Any
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp
from aiohttp import ClientError, ClientSession
from bs4 import BeautifulSoup

from ..models.document import Document, DocumentMetadata, DocumentType

logger = logging.getLogger(__name__)

# Common sitemap paths to check
COMMON_SITEMAP_PATHS = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap/sitemap.xml",
    "/sitemap-index.xml",
    "/sitemap",
]

# Common namespaces for sitemap XML
SITEMAP_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
    "video": "http://www.google.com/schemas/sitemap-video/1.1",
}


@dataclass
class SitemapUrl:
    """Represents a URL entry in a sitemap."""
    
    loc: str
    lastmod: Optional[str] = None
    changefreq: Optional[str] = None
    priority: Optional[float] = None
    images: List[Dict[str, str]] = None
    videos: List[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "loc": self.loc,
            "lastmod": self.lastmod,
            "changefreq": self.changefreq,
            "priority": self.priority,
            "images": self.images or [],
            "videos": self.videos or [],
        }


class SitemapCrawler:
    """Crawler for discovering and parsing sitemaps."""
    
    def __init__(
        self,
        base_url: str,
        session: Optional[ClientSession] = None,
        max_retries: int = 3,
        request_timeout: int = 30,
    ):
        """Initialize the sitemap crawler.
        
        Args:
            base_url: Base URL of the website to crawl
            session: Optional aiohttp ClientSession to reuse
            max_retries: Maximum number of retries for failed requests
            request_timeout: Timeout for HTTP requests in seconds
        """
        self.base_url = self.normalize_url(base_url)
        self.domain = urlparse(self.base_url).netloc
        self.session = session or ClientSession()
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.visited_urls: Set[str] = set()
        self.discovered_sitemaps: Set[str] = set()
        self.sitemap_urls: List[SitemapUrl] = []
    
    @staticmethod
    def normalize_url(url: str) -> str:
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
        """Discover sitemap files for the given base URL."""
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
        except (ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"Failed to fetch robots.txt: {e}")
        return []
    
    async def _check_sitemap_url(self, url: str) -> List[str]:
        """Check if a URL points to a valid sitemap."""
        try:
            async with self.session.head(url, timeout=self.request_timeout) as response:
                if response.status == 200 and \
                   response.headers.get("Content-Type", "").startswith(("application/xml", "text/xml")):
                    return [url]
        except (ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"Failed to check sitemap URL {url}: {e}")
        return []
    
    async def parse_sitemap(self, sitemap_url: str) -> List[SitemapUrl]:
        """Parse a sitemap URL and return the URLs it contains."""
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
                    return await self._parse_sitemap(content, sitemap_url)
                    
        except (ClientError, asyncio.TimeoutError) as e:
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
                    sitemap_url = self.normalize_url(loc_elem.text)
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
    
    async def _parse_sitemap(self, content: str, sitemap_url: str) -> List[SitemapUrl]:
        """Parse a standard sitemap and return all URLs."""
        try:
            root = ET.fromstring(content)
            urls = []
            
            for url_elem in root.findall(".//sm:url", SITEMAP_NS):
                loc_elem = url_elem.find("sm:loc", SITEMAP_NS)
                if loc_elem is None or not loc_elem.text:
                    continue
                    
                # Parse URL entry
                url = SitemapUrl(loc=self.normalize_url(loc_elem.text))
                
                # Parse optional fields
                lastmod_elem = url_elem.find("sm:lastmod", SITEMAP_NS)
                if lastmod_elem is not None and lastmod_elem.text:
                    url.lastmod = lastmod_elem.text
                    
                changefreq_elem = url_elem.find("sm:changefreq", SITEMAP_NS)
                if changefreq_elem is not None and changefreq_elem.text:
                    url.changefreq = changefreq_elem.text
                    
                priority_elem = url_elem.find("sm:priority", SITEMAP_NS)
                if priority_elem is not None and priority_elem.text:
                    try:
                        url.priority = float(priority_elem.text)
                    except (ValueError, TypeError):
                        pass
                
                # Parse images if present
                url.images = []
                for img_elem in url_elem.findall(".//image:image", SITEMAP_NS):
                    img_loc = img_elem.find("image:loc", SITEMAP_NS)
                    if img_loc is not None and img_loc.text:
                        img = {"loc": img_loc.text}
                        
                        # Add optional image attributes
                        for attr in ["caption", "geo_location", "title", "license"]:
                            elem = img_elem.find(f"image:{attr}", SITEMAP_NS)
                            if elem is not None and elem.text:
                                img[attr] = elem.text
                                
                        url.images.append(img)
                
                # Parse videos if present
                url.videos = []
                for video_elem in url_elem.findall(".//video:video", SITEMAP_NS):
                    video = {}
                    
                    # Add video attributes
                    for attr in ["thumbnail_loc", "title", "description", "content_loc", "player_loc"]:
                        elem = video_elem.find(f"video:{attr}", SITEMAP_NS)
                        if elem is not None and elem.text:
                            video[attr] = elem.text
                    
                    # Add duration if present
                    duration_elem = video_elem.find("video:duration", SITEMAP_NS)
                    if duration_elem is not None and duration_elem.text:
                        try:
                            video["duration"] = int(duration_elem.text)
                        except (ValueError, TypeError):
                            pass
                            
                    if video:
                        url.videos.append(video)
                
                urls.append(url)
            
            return urls
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap {sitemap_url}: {e}")
            return []
    
    async def crawl(self) -> List[SitemapUrl]:
        """Crawl and parse all discovered sitemaps."""
        # First discover sitemaps
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
    
    async def close(self):
        """Close the HTTP session if it was created by this instance."""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            await self.session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


async def extract_sitemap_urls(base_url: str) -> List[Dict[str, Any]]:
    """Extract URLs from a website's sitemap.
    
    Args:
        base_url: Base URL of the website
        
    Returns:
        List of dictionaries containing URL information
    """
    async with SitemapCrawler(base_url) as crawler:
        urls = await crawler.crawl()
        return [url.to_dict() for url in urls]


async def crawl_sitemap_to_documents(
    base_url: str,
    include_images: bool = True,
    include_videos: bool = False,
) -> List[Document]:
    """Crawl a sitemap and return Document objects.
    
    Args:
        base_url: Base URL of the website
        include_images: Whether to include image metadata
        include_videos: Whether to include video metadata
        
    Returns:
        List of Document objects
    """
    async with SitemapCrawler(base_url) as crawler:
        urls = await crawler.crawl()
        
        documents = []
        for url in urls:
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
            
            if include_images and url.images:
                additional_metadata["images"] = url.images
                
            if include_videos and url.videos:
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
