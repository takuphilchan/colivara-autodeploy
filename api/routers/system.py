"""
System Router - Health check and API information endpoints
"""

from fastapi import APIRouter, Depends
from datetime import datetime
import httpx
import logging

from api.core.config import settings
from api.core.dependencies import verify_api_key
from api.schemas import HealthResponse, SystemInfoResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Checks the status of the API and its dependencies (ColiVara).
    """
    try:
        # Test connection to ColiVara
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.COLIVARA_BASE_URL}/")
            colivara_status = "healthy" if response.status_code < 500 else "unhealthy"
    except Exception as e:
        logger.warning(f"ColiVara health check failed: {e}")
        colivara_status = "unreachable"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "healthy",
            "colivara": colivara_status,
        },
        "version": "2.0.0"
    }


@router.get("/", response_model=SystemInfoResponse, tags=["Root"])
async def root():
    """
    Root endpoint with API information
    
    Returns comprehensive information about the API including:
    - Available endpoints
    - Authentication methods
    - Configuration details
    - Supported file types
    - Usage examples
    """
    return {
        "message": "Document Q&A API with ColiVara Integration",
        "version": "2.0.0",
        "endpoints": {
            "query": "/query (POST) - Search across documents using a question",
            "upload": "/upload (POST) - Upload single document",
            "upload_bulk": "/upload-bulk (POST) - Bulk upload from folder",
            "search_documents": "/search-documents (GET) - Search documents by partial name match",
            "list_documents": "/documents (GET) - List all documents in collection",
            "delete": "/delete/{filename} (DELETE) - Delete document by exact filename",
            "delete_matching": "/delete-matching (DELETE) - Bulk delete by matching keyword in filename",
            "preview": "/preview/{document_name} (GET) - Preview document pages",
            "document_exists": "/document-exists/{document_name} (GET) - Check if document exists",
            "upload_status": "/upload-status/{document_name} (GET) - Check upload status",
            "health": "/health (GET) - API health check"
        },
        "authentication": {
            "methods": ["Query parameter (api_key)", "Bearer token (Authorization header)"],
            "note": "All endpoints require valid API key"
        },
        "configurations": {
            "colivara_base_url": settings.COLIVARA_BASE_URL,
            "minio_bucket": settings.MINIO_BUCKET,
            "default_collection": settings.DEFAULT_COLLECTION,
            "ollama_url": settings.OLLAMA_URL,
            "model_name": settings.MODEL_NAME
        },
        "supported_file_types": [".pdf", ".doc", ".docx", ".txt"],
        "examples": {
            "list_documents": f"GET /documents?api_key=YOUR_KEY&collection_name={settings.DEFAULT_COLLECTION}",
            "delete_document": f"DELETE /delete/document.pdf?api_key=YOUR_KEY&collection_name={settings.DEFAULT_COLLECTION}",
            "delete_matching": f"DELETE /delete-matching?keyword=_1&api_key=YOUR_KEY&collection_name={settings.DEFAULT_COLLECTION}",
            "search_documents": f"GET /search-documents?query=invoice&api_key=YOUR_KEY&collection_name={settings.DEFAULT_COLLECTION}",
            "upload_document": f"POST /upload with form data: file, api_key, collection_name"
        }
    }
