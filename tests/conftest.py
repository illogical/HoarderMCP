"""Pytest configuration and fixtures for HoarderMCP tests."""
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional, Dict, Any
from unittest.mock import MagicMock, AsyncMock

import pytest
import pytest_asyncio
from aiohttp import ClientSession

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for tests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Reduce log noise for test output
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"


def pytest_configure(config):
    """Pytest configuration hook."""
    # Create test data directory if it doesn't exist
    TEST_DATA_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for each test case.

    This is necessary because the default event loop is closed after each test.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def http_session() -> AsyncGenerator[ClientSession, None]:
    """Create an aiohttp client session for testing."""
    async with ClientSession() as session:
        yield session


@pytest.fixture
def mock_http_response():
    """Create a mock HTTP response for testing."""
    def _create_mock_response(
        status: int = 200,
        content_type: str = "application/json",
        text: str = "",
        json_data: dict = None,
        headers: dict = None,
    ) -> MagicMock:
        """Create a mock response with the given parameters."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.content_type = content_type
        
        # Mock text() coroutine
        mock_resp.text = AsyncMock(return_value=text)
        
        # Mock json() coroutine if json_data is provided
        if json_data is not None:
            mock_resp.json = AsyncMock(return_value=json_data)
        
        # Set headers
        mock_resp.headers = headers or {}
        
        return mock_resp
    
    return _create_mock_response


@pytest.fixture
def mock_aioresponse():
    """Create a mock aiohttp response using aioresponses."""
    try:
        from aioresponses import aioresponses
        with aioresponses() as m:
            yield m
    except ImportError:
        pytest.skip("aioresponses is not installed")
