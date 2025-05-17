# HoarderMCP

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

HoarderMCP is a Model Context Protocol (MCP) server designed for web content crawling, processing, and vector storage. It provides tools for ingesting web content, extracting relevant information, and making it searchable through vector similarity search.

## Features

- **Web Crawling**: Crawl websites and sitemaps to extract content
- **Advanced Content Processing**:
  - Semantic chunking for Markdown with header-based splitting
  - Code-aware chunking for Python and C# with syntax preservation
  - Configurable chunk sizes and overlap for optimal context
  - Token-based size optimization
- **Vector Storage**: Store and search content using vector embeddings
- **API-First**: RESTful API for easy integration with other services
- **Asynchronous**: Built with async/await for high performance
- **Extensible**: Support for multiple vector stores (Milvus, FAISS, Chroma, etc.)
- **Observability**: Integrated with Langfuse for tracing and monitoring

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hoardermcp.git
   cd hoardermcp
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Prerequisites

- Python 3.9+
- [Milvus](https://milvus.io/docs/install_standalone-docker.md) (or another supported vector store)
- [Docker](https://www.docker.com/) (for running Milvus)

## Running the Server

1. Start Milvus using Docker:
   ```bash
   docker-compose up -d
   ```

2. Run the development server:
   ```bash
   python -m hoardermcp.main --reload
   ```

   The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- **OpenAPI docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

Configuration can be provided through environment variables or a `.env` file. See `.env.example` for available options.

## Usage

### Ingesting Content

```python
import httpx

# Ingest a webpage
response = httpx.post(
    "http://localhost:8000/ingest",
    json={
        "sources": [
            {
                "url": "https://example.com",
                "content_type": "text/html"
            }
        ]
    }
)
print(response.json())
```

### Advanced Chunking Example

```python
from hoardermcp.core.chunking import ChunkingFactory, ChunkingConfig, ChunkingStrategy
from hoardermcp.models.document import Document, DocumentMetadata, DocumentType

# Create a markdown document
markdown_content = """# Title\n\n## Section 1\nContent for section 1\n\n## Section 2\nContent for section 2"""
doc = Document(
    id="test",
    content=markdown_content,
    metadata=DocumentMetadata(
        source="example.md",
        content_type=DocumentType.MARKDOWN
    )
)

# Configure chunking
config = ChunkingConfig(
    strategy=ChunkingStrategy.SEMANTIC,
    chunk_size=1000,
    chunk_overlap=200
)

# Get appropriate chunker and process document
chunker = ChunkingFactory.get_chunker(
    doc_type=DocumentType.MARKDOWN,
    config=config
)
chunks = chunker.chunk_document(doc)

print(f"Document split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1} (length: {len(chunk.content)}): {chunk.content[:50]}...")
```

### Searching Content

```python
import httpx

# Search for similar content
response = httpx.post(
    "http://localhost:8000/search",
    json={
        "query": "What is HoarderMCP?",
        "k": 5
    }
)
print(response.json())
```

## Development

### Code Style

This project uses:

- [Black](https://github.com/psf/black) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [mypy](http://mypy-lang.org/) for static type checking

Run the following commands before committing:

```bash
black .
isort .
mypy .
```

### Testing

Run tests using pytest:

```bash
pytest
```
