"""
Documents Router - API endpoints for document management
Handles upload, deletion, listing, and preview operations
"""

from fastapi import APIRouter, UploadFile, File, Form, Query, Path, Depends
from typing import Optional
import logging

from api.services.document_service import DocumentService
from api.core.dependencies import verify_api_key
from api.core.config import settings
from api.schemas import (
    UploadResponse,
    DocumentListResponse,
    DocumentSearchResponse,
    DeleteResponse,
    DeleteMatchingResponse,
    DocumentExistsResponse,
    UploadStatusResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()
service = DocumentService()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    api_key: str = Form(default=settings.API_KEY, description="API key for authentication"),
    collection_name: str = Form(default=settings.DEFAULT_COLLECTION, description="Collection to upload to"),
    wait_for_processing: bool = Form(default=True, description="Wait for document processing to complete")
):
    """
    Upload a single document to ColiVara
    
    **Features:**
    - Automatic duplicate detection
    - Comprehensive metadata
    - Progress tracking
    - Support for multiple file types
    
    **Supported file types:**
    - PDF, DOC, DOCX, TXT
    - Images: PNG, JPG, JPEG, TIFF, BMP
    - Spreadsheets: CSV, XLS, XLSX
    - Presentations: PPT, PPTX
    - Others: MD, HTML, JSON
    """
    result = await service.upload_document(file, api_key, collection_name, wait_for_processing)
    return UploadResponse(**result)


@router.post("/upload-bulk", response_model=dict)
async def upload_bulk(
    folder_path: str = Form(..., description="Path to folder containing documents"),
    api_key: str = Form(default=settings.API_KEY, description="API key for authentication"),
    collection_name: str = Form(default=settings.DEFAULT_COLLECTION, description="Collection to upload to"),
    max_file_size_mb: int = Form(default=50, description="Maximum file size in MB", ge=1, le=500),
    continue_on_error: bool = Form(default=True, description="Continue uploading if individual files fail"),
    max_concurrent: int = Form(default=3, description="Maximum concurrent uploads", ge=1, le=10)
):
    """
    Bulk upload documents from a folder
    
    **Features:**
    - Concurrent uploads for better performance
    - Automatic duplicate detection
    - Size validation
    - Detailed progress reporting
    - Error handling per file
    
    **Response includes:**
    - Total files processed
    - Successful uploads
    - Failed uploads with error messages
    - Skipped files (existing, too large, invalid format)
    """
    return await service.bulk_upload(
        folder_path=folder_path,
        api_key=api_key,
        collection_name=collection_name,
        max_file_size_mb=max_file_size_mb,
        continue_on_error=continue_on_error,
        max_concurrent=max_concurrent
    )


@router.get("/upload-status/{document_name}", response_model=dict)
async def check_upload_status(
    document_name: str = Path(..., description="Document name to check status for"),
    api_key: str = Query(default=settings.API_KEY, description="API key for authentication"),
    collection_name: str = Query(default=settings.DEFAULT_COLLECTION, description="Collection name")
):
    """
    Check upload status of a document
    
    Returns information about whether a document exists in the collection
    and its processing status.
    """
    import urllib.parse
    
    decoded_name = urllib.parse.unquote(document_name)
    logger.info(f"Checking upload status for: {decoded_name}")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Check if document exists
        existing_docs = await service.get_existing_documents(headers, collection_name)
        
        if decoded_name in existing_docs:
            # Get document info
            doc_info = await service.get_document_info(headers, collection_name, decoded_name)
            
            return {
                "status": "completed",
                "document_name": decoded_name,
                "collection_name": collection_name,
                "exists": True,
                "document_id": doc_info.get("id") if doc_info else None,
                "metadata": doc_info.get("metadata") if doc_info else None
            }
        else:
            return {
                "status": "not_found",
                "document_name": decoded_name,
                "collection_name": collection_name,
                "exists": False
            }
    
    except Exception as e:
        logger.error(f"Error checking upload status: {str(e)}")
        return {
            "status": "error",
            "document_name": decoded_name,
            "error": str(e)
        }


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    api_key: str = Query(default=settings.API_KEY, description="API key for authentication"),
    collection_name: str = Query(default=settings.DEFAULT_COLLECTION, description="Collection name")
):
    """
    List all documents in a collection
    
    Returns comprehensive information about all documents including:
    - Document name
    - File size
    - Upload metadata
    - Processing status
    """
    result = await service.list_documents(api_key, collection_name)
    return DocumentListResponse(**result)


@router.get("/search-documents", response_model=dict)
async def search_documents(
    query: str = Query(..., description="Search query for document names"),
    api_key: str = Query(default=settings.API_KEY, description="API key for authentication"),
    collection_name: str = Query(default=settings.DEFAULT_COLLECTION, description="Collection name")
):
    """
    Search documents by name
    
    Performs a case-insensitive substring search on document names.
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Get all documents
        all_docs = await service.get_existing_documents(headers, collection_name)
        
        # Filter by query
        query_lower = query.lower()
        matching_docs = [
            {"name": doc}
            for doc in all_docs
            if query_lower in doc.lower()
        ]
        
        return {
            "documents": matching_docs,
            "total_count": len(matching_docs),
            "search_term": query,
            "collection_name": collection_name
        }
    
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise


@router.get("/preview/{document_name}", response_model=dict)
async def preview_document(
    document_name: str = Path(..., description="Document name to preview"),
    page_number: Optional[int] = Query(default=None, ge=1, description="Specific page number (optional)"),
    max_pages: int = Query(default=1, ge=1, le=10, description="Maximum number of pages to return"),
    api_key: str = Query(default=settings.API_KEY, description="API key for authentication"),
    collection_name: str = Query(default=settings.DEFAULT_COLLECTION, description="Collection name")
):
    """
    Preview document pages
    
    Get visual and text preview of document pages.
    
    **Note:** This endpoint requires ColiVara search integration
    which will be implemented when the query service is complete.
    """
    import urllib.parse
    
    decoded_name = urllib.parse.unquote(document_name)
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Check if document exists
        existing_docs = await service.get_existing_documents(headers, collection_name)
        
        if decoded_name not in existing_docs:
            return {
                "error": "Document not found",
                "document_name": decoded_name,
                "collection_name": collection_name,
                "exists": False
            }
        
        # For now, return basic info
        # Full preview functionality will be added with query service
        return {
            "document_name": decoded_name,
            "collection_name": collection_name,
            "exists": True,
            "preview_mode": "basic",
            "message": "Full preview functionality coming soon with query service integration"
        }
    
    except Exception as e:
        logger.error(f"Error previewing document: {str(e)}")
        raise


@router.get("/document-exists/{document_name}", response_model=dict)
async def check_document_exists(
    document_name: str = Path(..., description="Document name to check"),
    api_key: str = Query(default=settings.API_KEY, description="API key for authentication"),
    collection_name: str = Query(default=settings.DEFAULT_COLLECTION, description="Collection name")
):
    """
    Check if a document exists in the collection
    
    Performs enhanced matching including:
    - Exact match
    - Case-insensitive match
    - Normalized matching (ignoring separators)
    """
    import urllib.parse
    
    decoded_name = urllib.parse.unquote(document_name)
    clean_name = decoded_name.strip()
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Get existing documents
        existing_docs = await service.get_existing_documents(headers, collection_name)
        
        # Enhanced matching
        exists = False
        matched_name = None
        
        # Direct match
        if clean_name in existing_docs:
            exists = True
            matched_name = clean_name
        else:
            # Case-insensitive matching
            for existing_doc in existing_docs:
                if existing_doc.lower() == clean_name.lower():
                    exists = True
                    matched_name = existing_doc
                    break
            
            # Normalized matching
            if not exists:
                for existing_doc in existing_docs:
                    clean_existing = existing_doc.replace('_', '').replace('-', '').replace(' ', '').lower()
                    clean_target = clean_name.replace('_', '').replace('-', '').replace(' ', '').lower()
                    if clean_existing == clean_target:
                        exists = True
                        matched_name = existing_doc
                        break
        
        return {
            "document_name": clean_name,
            "matched_document_name": matched_name,
            "collection_name": collection_name,
            "exists": exists,
            "total_documents_in_collection": len(existing_docs)
        }
    
    except Exception as e:
        logger.error(f"Error checking document existence: {str(e)}")
        raise


@router.delete("/delete/{filename}", response_model=DeleteResponse)
async def delete_document(
    filename: str = Path(..., description="Document filename to delete"),
    api_key: str = Query(default=settings.API_KEY, description="API key for authentication"),
    collection_name: str = Query(default=settings.DEFAULT_COLLECTION, description="Collection name")
):
    """
    Delete a single document by exact filename
    
    **Note:** Filename must match exactly (case-sensitive)
    """
    result = await service.delete_document(filename, api_key, collection_name)
    return DeleteResponse(**result)


@router.delete("/delete-matching", response_model=dict)
async def delete_matching_documents(
    keyword: str = Query(..., description="Substring to match in document names"),
    api_key: str = Query(default=settings.API_KEY, description="API key for authentication"),
    collection_name: str = Query(default=settings.DEFAULT_COLLECTION, description="Collection name")
):
    """
    Delete all documents that contain a specific keyword in their filename
    
    **Warning:** This operation cannot be undone. Use with caution.
    
    **Features:**
    - Case-insensitive matching
    - Detailed results for each deletion
    - Continues on individual failures
    
    **Example:**
    - keyword="_1" will delete all files containing "_1" in their name
    - keyword="invoice" will delete all files containing "invoice"
    """
    return await service.delete_matching_documents(keyword, api_key, collection_name)
