"""Main entry point for the HoarderMCP server."""
import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from . import __version__
from .api.server import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="HoarderMCP - Web Content Crawling and Vector Storage MCP Server")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=bool(os.getenv("RELOAD", "False").lower() == "true"),
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "1")),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"HoarderMCP {__version__}",
        help="Show version and exit",
    )
    return parser.parse_args()


def main():
    """Run the FastAPI application."""
    # Load environment variables from .env file if it exists
    env_path = Path(".") / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loaded environment variables from {env_path}")

    # Parse command line arguments
    args = parse_args()

    # Set log level
    logging.basicConfig(level=args.log_level.upper())

    # Configure Uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = (
        "%(asctime)s - %(name)s - %(levelname)s - %(client_addr)s - \"%(request_line)s\" %(status_code)s"
    )

    # Start the server
    uvicorn.run(
        "hoardermcp.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
