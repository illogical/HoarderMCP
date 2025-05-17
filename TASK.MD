# HoarderMCP Project Tasks

## Completed Tasks (2024-05-17)

### Core Infrastructure
- [x] Set up project structure with proper Python packaging
- [x] Configured development environment with Docker and Docker Compose
- [x] Implemented Pydantic models for documents, chunks, and metadata
- [x] Set up Milvus vector store integration
- [x] Created base vector store abstraction layer
- [x] Implemented document ingestion service with chunking
- [x] Added web crawler with Crawl4AI integration
- [x] Created FastAPI server with RESTful endpoints
- [x] Added background task processing for long-running operations
- [x] Set up configuration management with environment variables
- [x] Implemented health checks and versioning
- [x] Added CORS middleware and error handling

### Documentation
- [x] Created comprehensive README with setup instructions
- [x] Added API documentation via OpenAPI/Swagger UI
- [x] Created CONTRIBUTING.md and CODE_OF_CONDUCT.md
- [x] Added example .env file
- [x] Set up development dependencies and tooling

## In Progress

### Core Features
- [x] Implement sitemap crawling functionality
  - [x] Create SitemapCrawler class with async support
  - [x] Add support for sitemap indexes and nested sitemaps
  - [x] Implement discovery of sitemaps via common locations and robots.txt
  - [x] Add metadata extraction from sitemap entries
  - [x] Integrate with main Crawler class
  - [x] Add comprehensive error handling and retries
    - [x] Implement exponential backoff with jitter
    - [x] Handle rate limiting and server errors
    - [x] Add proper timeout and connection error handling
  - [x] Add tests for sitemap crawling
    - [x] Unit tests for SitemapCrawler class
    - [x] Test error handling and retry logic
    - [x] Test sitemap discovery and parsing
    - [x] Test document conversion
- [x] Simplify deployment and dependencies to a single dockerfile for the API and MCP server
  - [x] Create a multi-stage Dockerfile for production
  - [x] Set up Docker Compose with all dependencies (Milvus, etcd, MinIO)
  - [x] Add startup and shutdown scripts
  - [x] Create deployment documentation

### Content Processing
- [x] Implement better chunking strategies for code and markdown
  - [x] Add semantic chunking for markdown with header-based splitting
  - [x] Implement code-aware chunking for programming languages (Python, C#)
  - [x] Add overlap between chunks for better context
  - [x] Implement chunk size optimization based on token count
  - [x] Add tests for chunking strategies
  - [x] Integrate chunking with the ingestion service

### Documentation
- [ ] Create usage examples and tutorials
- [ ] Add architecture decision records (ADRs)
- [ ] Document deployment strategies

## Future Enhancements

### Features
- [ ] Add web UI for monitoring and management
- [ ] Add support for custom extractors and processors
- [ ] Implement caching layer with Redis via Docker
- [ ] Implement user authentication and API keys
- [ ] Implement duplicate content detection
- [ ] Add rate limiting and request throttling
- [ ] Add support for additional vector stores (FAISS, Chroma)

### Performance
- [ ] Implement batch processing for large document sets
- [ ] Add distributed processing support
- [ ] Optimize vector search performance
- [ ] Implement caching for frequently accessed content

### Observability
- [ ] Add Langfuse metrics
- [ ] Implement distributed tracing
- [ ] Set up alerting for system health

### Testing
- [ ] Write unit tests for core functionality
- [ ] Add integration tests for API endpoints
- [ ] Set up CI/CD pipeline
- [ ] Add performance benchmarking

## Stretch Goals

### Features
- [ ] Implement scheduled crawling
  - [ ] Create Scheduler class with async support
  - [ ] Add support for cron-style scheduling
  - [ ] Implement persistent job storage
  - [ ] Add API endpoints for managing scheduled jobs
  - [ ] Add validation for schedule patterns
  - [ ] Implement job status tracking and history
  - [ ] Add error handling and retries for failed jobs
  - [ ] Add tests for scheduler functionality

### Content Processing
- [ ] Implement incremental updates for crawled content
  - [ ] Add last_modified tracking for crawled content
  - [ ] Implement change detection for updated content
  - [ ] Add support for ETags and If-Modified-Since headers
  - [ ] Create diffing mechanism for partial updates
  - [ ] Add incremental update scheduling
  - [ ] Implement cleanup of stale content
  - [ ] Add tests for incremental update functionality

## Notes
- Follow the established code style and documentation standards
- Update documentation when adding new features or changing behavior
- Consider backward compatibility when making changes to the API

## Discovered During Work
- Add more detailed error messages for API consumers
- Implement retry logic for failed operations
- Add support for content versioning