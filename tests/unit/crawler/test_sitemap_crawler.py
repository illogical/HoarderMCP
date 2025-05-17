"""Unit tests for the SitemapCrawler class."""
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from unittest import mock

import aiohttp
import pytest
import xml.etree.ElementTree as ET
from unittest.mock import AsyncMock, MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hoardermcp.crawler.sitemap import SitemapCrawler, SitemapUrl, SITEMAP_NS

# Sample test data
SAMPLE_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml"
        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1"
        xmlns:video="http://www.google.com/schemas/sitemap-video/1.1">
  <url>
    <loc>https://example.com/page1</loc>
    <lastmod>2023-01-01</lastmod>
    <changefreq>daily</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://example.com/page2</loc>
    <lastmod>2023-01-02</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.5</priority>
  </url>
</urlset>"""

SAMPLE_SITEMAP_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://example.com/sitemap1.xml</loc>
    <lastmod>2023-01-01</lastmod>
  </sitemap>
  <sitemap>
    <loc>https://example.com/sitemap2.xml</loc>
    <lastmod>2023-01-02</lastmod>
  </sitemap>
</sitemapindex>"""

SAMPLE_ROBOTS_TXT = """User-agent: *
Allow: /
Sitemap: https://example.com/sitemap.xml
Sitemap: https://example.com/sitemap_news.xml
"""


@pytest.fixture
def mock_session():
    """Create a mock aiohttp.ClientSession."""
    with patch('aiohttp.ClientSession') as mock_session_cls:
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_cls.return_value = mock_session
        yield mock_session


@pytest.fixture
def sitemap_crawler():
    """Create a SitemapCrawler instance for testing."""
    return SitemapCrawler("https://example.com")


@pytest.mark.asyncio
async def test_discover_sitemaps(mock_session, sitemap_crawler):
    """Test discovering sitemaps from common locations and robots.txt."""
    # Mock responses for common sitemap locations
    mock_responses = {
        "https://example.com/sitemap.xml": (200, "text/xml", "<urlset></urlset>"),
        "https://example.com/robots.txt": (200, "text/plain", SAMPLE_ROBOTS_TXT),
    }
    
    async def mock_get(url, *args, **kwargs):
        status, content_type, body = mock_responses.get(url, (404, "text/plain", "Not Found"))
        response = AsyncMock()
        response.status = status
        response.headers = {"Content-Type": content_type}
        response.text.return_value = body
        return response
    
    mock_session.head.side_effect = mock_get
    mock_session.get.side_effect = mock_get
    
    sitemaps = await sitemap_crawler.discover_sitemaps()
    
    assert len(sitemaps) == 3  # 1 from common location + 2 from robots.txt
    assert "https://example.com/sitemap.xml" in sitemaps
    assert "https://example.com/sitemap_news.xml" in sitemaps


@pytest.mark.asyncio
async def test_parse_sitemap(sitemap_crawler):
    """Test parsing a sitemap XML."""
    with patch.object(sitemap_crawler, '_make_request') as mock_request:
        # Mock the response for _make_request
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = SAMPLE_SITEMAP_XML
        mock_request.return_value = mock_response
        
        urls = await sitemap_crawler.parse_sitemap("https://example.com/sitemap.xml")
        
        assert len(urls) == 2
        assert urls[0].loc == "https://example.com/page1"
        assert urls[0].lastmod == "2023-01-01"
        assert urls[0].changefreq == "daily"
        assert urls[0].priority == 0.8


@pytest.mark.asyncio
async def test_parse_sitemap_index(sitemap_crawler):
    """Test parsing a sitemap index."""
    with patch.object(sitemap_crawler, '_make_request') as mock_request, \
         patch.object(sitemap_crawler, 'parse_sitemap') as mock_parse_sitemap:
        
        # Mock the response for the index
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = SAMPLE_SITEMAP_INDEX
        mock_request.return_value = mock_response
        
        # Mock the response for the individual sitemaps
        mock_parse_sitemap.side_effect = [
            [SitemapUrl(loc="https://example.com/page1")],
            [SitemapUrl(loc="https://example.com/page2")]
        ]
        
        urls = await sitemap_crawler.parse_sitemap("https://example.com/sitemap_index.xml")
        
        assert len(urls) == 2
        assert urls[0].loc == "https://example.com/page1"
        assert urls[1].loc == "https://example.com/page2"


@pytest.mark.asyncio
async def test_crawl(sitemap_crawler):
    """Test the full crawl process."""
    with patch.object(sitemap_crawler, 'discover_sitemaps') as mock_discover, \
         patch.object(sitemap_crawler, 'parse_sitemap') as mock_parse:
        
        # Mock discovering sitemaps
        mock_discover.return_value = ["https://example.com/sitemap1.xml"]
        
        # Mock parsing the sitemap
        mock_parse.return_value = [
            SitemapUrl(
                loc="https://example.com/page1",
                lastmod="2023-01-01",
                changefreq="daily",
                priority=0.8
            )
        ]
        
        urls = await sitemap_crawler.crawl()
        
        assert len(urls) == 1
        assert urls[0].loc == "https://example.com/page1"
        mock_discover.assert_called_once()
        mock_parse.assert_called_once_with("https://example.com/sitemap1.xml")


@pytest.mark.asyncio
async def test_make_request_retry(sitemap_crawler):
    """Test the retry logic in _make_request."""
    with patch('aiohttp.ClientSession.request') as mock_request:
        # First two requests fail, third succeeds
        mock_request.side_effect = [
            aiohttp.ClientError("Connection error"),
            aiohttp.ClientError("Timeout"),
            MagicMock(status=200, headers={"Content-Type": "text/plain"}, text=AsyncMock(return_value="Success")),
        ]
        
        response = await sitemap_crawler._make_request("https://example.com")
        
        assert response is not None
        assert mock_request.call_count == 3


@pytest.mark.asyncio
async def test_to_documents(sitemap_crawler):
    """Test converting sitemap URLs to documents."""
    # Add some test URLs to the crawler
    sitemap_crawler.sitemap_urls = [
        SitemapUrl(
            loc="https://example.com/page1",
            lastmod="2023-01-01",
            changefreq="daily",
            priority=0.8,
            images=["https://example.com/image1.jpg"]
        )
    ]
    
    documents = await sitemap_crawler.to_documents()
    
    assert len(documents) == 1
    assert documents[0].id == "https://example.com/page1"
    assert documents[0].metadata.source == "https://example.com/page1"
    assert documents[0].metadata.last_modified == "2023-01-01"
    assert documents[0].metadata.change_frequency == "daily"
    assert documents[0].metadata.priority == 0.8
    assert documents[0].metadata.additional_metadata["images"] == ["https://example.com/image1.jpg"]
