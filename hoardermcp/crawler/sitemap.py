"""Sitemap crawler for HoarderMCP."""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import HttpUrl

from .crawler import Crawler, CrawlOptions, CrawlResult

logger = logging.getLogger(__name__)


@dataclass
class SitemapEntry:
    """A single entry in a sitemap."""

    loc: str
    "URL of the page"

    lastmod: Optional[datetime] = None
    "When the page was last modified"

    changefreq: Optional[str] = None
    "How frequently the page is likely to change"

    priority: Optional[float] = None
    "Priority of this URL relative to other URLs on the site"

    @classmethod
    def from_xml_element(cls, element: ET.Element) -> Optional[SitemapEntry]:
        """Create a SitemapEntry from an XML element.

        Args:
            element: XML element from a sitemap.

        Returns:
            SitemapEntry if the element is valid, None otherwise.
        """
        # Sitemap namespaces
        ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Extract URL
        loc_elem = element.find("ns:loc", ns)
        if loc_elem is None or not loc_elem.text:
            return None

        entry = cls(loc=loc_elem.text.strip())

        # Extract last modification date
        lastmod_elem = element.find("ns:lastmod", ns)
        if lastmod_elem is not None and lastmod_elem.text:
            try:
                entry.lastmod = datetime.fromisoformat(
                    lastmod_elem.text.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        # Extract change frequency
        changefreq_elem = element.find("ns:changefreq", ns)
        if changefreq_elem is not None and changefreq_elem.text:
            entry.changefreq = changefreq_elem.text.lower().strip()

        # Extract priority
        priority_elem = element.find("ns:priority", ns)
        if priority_elem is not None and priority_elem.text:
            try:
                entry.priority = float(priority_elem.text)
            except (ValueError, TypeError):
                pass

        return entry

    def to_dict(self) -> Dict:
        """Convert to a dictionary.

        Returns:
            Dictionary representation of the sitemap entry.
        """
        return {
            "loc": self.loc,
            "lastmod": self.lastmod.isoformat() if self.lastmod else None,
            "changefreq": self.changefreq,
            "priority": self.priority,
        }


class SitemapCrawler:
    """Crawler for sitemap.xml files."""

    def __init__(self, base_url: Optional[str] = None):
        """Initialize the sitemap crawler.

        Args:
            base_url: Base URL of the website (used for relative URLs).
        """
        self.base_url = base_url
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
