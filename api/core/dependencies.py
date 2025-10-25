"""
Dependency injection and shared resources.
Provides common dependencies for FastAPI routes.
"""
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from minio import Minio
from colivara_py import ColiVara
from typing import Optional
import logging

from api.core.config import settings

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global clients (initialized once)
_minio_client: Optional[Minio] = None
_colivara_client: Optional[ColiVara] = None


def get_minio_client() -> Minio:
    """Get or create MinIO client instance"""
    global _minio_client
    
    if _minio_client is None:
        try:
            _minio_client = Minio(**settings.get_minio_config())
            logger.info(f"MinIO client initialized: {settings.MINIO_ENDPOINT}")
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            raise HTTPException(status_code=500, detail="Storage service unavailable")
    
    return _minio_client


def get_colivara_client() -> ColiVara:
    """Get or create ColiVara client instance"""
    global _colivara_client
    
    if _colivara_client is None:
        try:
            _colivara_client = ColiVara(
                base_url=settings.COLIVARA_BASE_URL,
                api_key=settings.API_KEY
            )
            logger.info(f"ColiVara client initialized: {settings.COLIVARA_BASE_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize ColiVara client: {e}")
            raise HTTPException(status_code=500, detail="RAG service unavailable")
    
    return _colivara_client


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    Verify API key from Authorization header.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    api_key = credentials.credentials
    
    if api_key != settings.API_KEY:
        logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key


def get_api_key(api_key: Optional[str] = None) -> str:
    """
    Get API key from parameter or use default.
    
    Args:
        api_key: Optional API key parameter
        
    Returns:
        API key to use
    """
    return api_key or settings.API_KEY


class CommonQueryParams:
    """Common query parameters for document operations"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        collection_name: str = settings.DEFAULT_COLLECTION
    ):
        self.api_key = api_key or settings.API_KEY
        self.collection_name = collection_name
