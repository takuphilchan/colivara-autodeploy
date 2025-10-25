"""
FastAPI Application - Document Q&A API Server
Modern API with modular structure for document management and querying

Usage:
    python api_server.py
    
Or with uvicorn:
    uvicorn api_server:app --host 0.0.0.0 --port 5001 --reload
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path to avoid conflict with api.py file
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import re
from datetime import datetime

# Import core configuration (using absolute imports from api package)
from api.core.config import settings

# Import routers
from api.routers.system import router as system_router
from api.routers.documents import router as documents_router
from api.routers.query import router as query_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="Modern API for document management and querying with ColiVara",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security middleware - Block malicious requests
@app.middleware("http")
async def block_malicious_requests(request: Request, call_next):
    """Block common malicious path patterns"""
    MALICIOUS_PATTERNS = [
        r".*/(aspspy|aspxspy)\.(php|jsp|jspx|asp|aspx)",
        r".*(wp-admin|wp-login|xmlrpc\.php).*",
        r".*(\.env|\.git|\.svn).*"
    ]
    
    if any(re.match(pattern, request.url.path) for pattern in MALICIOUS_PATTERNS):
        return JSONResponse(
            status_code=403,
            content={"detail": "Forbidden"},
            headers={"Server": ""}
        )
    return await call_next(request)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.now()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Response: {response.status_code} | "
        f"Duration: {duration:.3f}s | "
        f"Path: {request.url.path}"
    )
    
    return response


# Register routers
app.include_router(system_router, tags=["System"])
app.include_router(documents_router, prefix="/api", tags=["Documents"])
app.include_router(query_router, prefix="/api", tags=["Query"])


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 60)
    logger.info("Starting Document Q&A API v2.0.0")
    logger.info("=" * 60)
    logger.info(f"ColiVara URL: {settings.COLIVARA_BASE_URL}")
    logger.info(f"Ollama URL: {settings.OLLAMA_URL}")
    logger.info(f"Model: {settings.MODEL_NAME}")
    logger.info(f"Default Collection: {settings.DEFAULT_COLLECTION}")
    minio_config = settings.get_minio_config()
    logger.info(f"MinIO Endpoint: {minio_config['endpoint']}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down Document Q&A API")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="info"
    )
