"""FastAPI server for HoarderMCP."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    status,
    Depends,
    Request,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from ..core import IngestService, VectorStoreService, ChunkingConfig, EmbeddingConfig
from ..models.document import Document, DocumentChunk, DocumentType
from ..models.ingest import IngestRequest, IngestSource, IngestStatus
from ..models.vector import VectorSearchRequest, VectorSearchResult, VectorStoreType

logger = logging.getLogger(__name__)

# Global application state
app_state: Dict[str, Any] = {}

# API version
API_VERSION = "0.1.0"


def get_ingest_service() -> IngestService:
    """Get the ingest service instance.

    Returns:
        IngestService instance.
    """
    if "ingest_service" not in app_state:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ingest service not initialized",
        )
    return app_state["ingest_service"]


def get_vector_store() -> VectorStoreService:
    """Get the vector store service instance.

    Returns:
        VectorStoreService instance.
    """
    if "vector_store" not in app_state:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector store not initialized",
        )
    return app_state["vector_store"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    Args:
        app: FastAPI application instance.
    """
    # Startup: Initialize services
    try:
        logger.info("Initializing services...")

        # Initialize vector store
        vector_store = VectorStoreService(
            config={
                "type": "milvus",
                "host": "localhost",
                "port": 19530,
                "collection_name": "hoardermcp_documents",
                "dimension": 384,  # all-MiniLM-L6-v2 dimension
            }
        )
        await vector_store.initialize()
        app_state["vector_store"] = vector_store

        # Initialize ingest service
        ingest_service = IngestService(
            vector_store=vector_store.store,
            chunking_config=ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=200,
            ),
            embedding_config=EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            ),
        )
        app_state["ingest_service"] = ingest_service

        logger.info("Services initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    finally:
        # Shutdown: Clean up resources
        logger.info("Shutting down services...")
        if "vector_store" in app_state:
            await app_state["vector_store"].close()
        logger.info("Services shut down successfully")


# Create FastAPI application
app = FastAPI(
    title="HoarderMCP",
    description="MCP server for web content crawling and vector storage",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions.

    Args:
        request: The request that caused the exception.
        exc: The HTTP exception.

    Returns:
        JSON response with error details.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other exceptions.

    Args:
        request: The request that caused the exception.
        exc: The exception.

    Returns:
        JSON response with error details.
    """
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# API endpoints
@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Status message.
    """
    return {"status": "ok"}


@app.get("/version")
async def get_version() -> Dict[str, str]:
    """Get the API version.

    Returns:
        API version information.
    """
    return {"version": API_VERSION}


class IngestResponse(BaseModel):
    """Response model for ingest operations."""

    job_id: str
    status: str
    message: str
    document_ids: List[str] = []


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    ingest_service: IngestService = Depends(get_ingest_service),
) -> IngestResponse:
    """Ingest documents from the provided sources.

    Args:
        request: Ingest request.
        background_tasks: Background tasks handler.
        ingest_service: Ingest service instance.

    Returns:
        Ingest response with job ID and status.
    """
    # Generate a unique job ID
    import uuid

    job_id = str(uuid.uuid4())

    # Start background task for ingestion
    background_tasks.add_task(
        process_ingestion, job_id, request, ingest_service
    )

    return IngestResponse(
        job_id=job_id,
        status=IngestStatus.PROCESSING,
        message="Ingestion started",
    )


async def process_ingestion(
    job_id: str, request: IngestRequest, ingest_service: IngestService
) -> None:
    """Process document ingestion in the background.

    Args:
        job_id: Job ID.
        request: Ingest request.
        ingest_service: Ingest service instance.
    """
    try:
        # TODO: Implement actual ingestion logic
        # This is a placeholder for the actual implementation
        logger.info(f"Starting ingestion job {job_id}")
        
        # Process each source in the request
        document_ids = []
        for source in request.sources:
            if source.url:
                # TODO: Handle URL sources
                logger.info(f"Processing URL: {source.url}")
            elif source.file_path:
                # TODO: Handle file paths
                logger.info(f"Processing file: {source.file_path}")
            elif source.content:
                # Handle direct content
                doc = Document(
                    id=str(uuid.uuid4()),
                    content=source.content,
                    metadata=DocumentMetadata(
                        source=source.url or "direct_content",
                        content_type=DocumentType.TEXT,
                    ),
                )
                result = await ingest_service.ingest_document(
                    content=doc.content,
                    metadata=doc.metadata.dict(),
                    content_type=doc.metadata.content_type,
                )
                document_ids.append(result.document_id)

        # Store the result in the app state
        app_state[f"ingest_result_{job_id}"] = {
            "status": IngestStatus.COMPLETED,
            "document_ids": document_ids,
            "message": f"Ingested {len(document_ids)} documents",
        }

        logger.info(f"Completed ingestion job {job_id}")

    except Exception as e:
        logger.error(f"Error in ingestion job {job_id}: {e}")
        app_state[f"ingest_result_{job_id}"] = {
            "status": IngestStatus.FAILED,
            "document_ids": [],
            "message": str(e),
        }


@app.get("/ingest/{job_id}", response_model=IngestResponse)
async def get_ingest_status(job_id: str) -> IngestResponse:
    """Get the status of an ingestion job.

    Args:
        job_id: Job ID.

    Returns:
        Ingest status response.
    """
    result = app_state.get(f"ingest_result_{job_id}")
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return IngestResponse(
        job_id=job_id,
        status=result["status"],
        message=result["message"],
        document_ids=result["document_ids"],
    )


@app.post("/search", response_model=List[VectorSearchResult])
async def search_documents(
    request: VectorSearchRequest,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> List[VectorSearchResult]:
    """Search for documents similar to the query.

    Args:
        request: Search request.
        vector_store: Vector store service instance.

    Returns:
        List of search results.
    """
    try:
        # TODO: Implement search with proper embedding
        # This is a placeholder for the actual implementation
        logger.info(f"Searching for: {request.query}")
        
        # For now, return empty results
        return []
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@app.get("/documents/{doc_id}", response_model=Document)
async def get_document(
    doc_id: str,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> Document:
    """Get a document by ID.

    Args:
        doc_id: Document ID.
        vector_store: Vector store service instance.

    Returns:
        Document if found.
    """
    document = await vector_store.get_document(doc_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found",
        )
    return document


@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> None:
    """Delete a document by ID.

    Args:
        doc_id: Document ID.
        vector_store: Vector store service instance.
    """
    success = await vector_store.delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found or could not be deleted",
        )


# Main entry point for running the server directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("hoardermcp.api.server:app", host="0.0.0.0", port=8000, reload=True)
