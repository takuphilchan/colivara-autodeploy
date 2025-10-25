"""
API Routers Package
"""

from fastapi import APIRouter

# Import all routers here
from .system import router as system_router
from .documents import router as documents_router
from .query import router as query_router

__all__ = [
    'system_router',
    'documents_router',
    'query_router',
]
