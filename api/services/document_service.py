"""
Document Service - Business logic for document management operations
Handles document upload, deletion, listing, and metadata operations
"""

import os
import base64
import logging
import asyncio
import mimetypes
import urllib.parse
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timezone, timedelta

import httpx
import aiofiles
from minio import Minio
from fastapi import HTTPException, UploadFile

from api.core.config import settings

logger = logging.getLogger(__name__)


class DocumentService:
    """Service class for document management operations"""
    
    def __init__(self):
        self.base_url = settings.COLIVARA_BASE_URL
        self.minio_client = None
        if hasattr(settings, 'get_minio_client'):
            try:
                self.minio_client = settings.get_minio_client()
            except Exception as e:
                logger.warning(f"Failed to initialize MinIO client: {e}")
    
    async def get_existing_documents(self, headers: dict, collection_name: str) -> Set[str]:
        """
        Retrieve existing documents from collection
        
        Args:
            headers: Authentication headers
            collection_name: Name of the collection
            
        Returns:
            Set of document names in the collection
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/documents/",
                    params={"collection_name": collection_name},
                    headers=headers,
                    timeout=15
                )
                response.raise_for_status()
                documents = response.json()
                return {doc['name'] for doc in documents}
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return set()
    
    async def get_document_info(self, headers: dict, collection_name: str, filename: str) -> Optional[Dict]:
        """
        Get full document information
        
        Args:
            headers: Authentication headers
            collection_name: Name of the collection
            filename: Document filename
            
        Returns:
            Document information dict or None if not found
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.base_url}/documents/",
                    headers=headers,
                    params={"collection_name": collection_name}
                )
                resp.raise_for_status()
                for doc in resp.json():
                    if doc.get("name") == filename:
                        return doc
            return None
        except Exception as e:
            logger.error(f"Error getting document info for {filename}: {e}")
            return None
    
    def get_system_info(self) -> Dict[str, str]:
        """Get system information for metadata"""
        import platform
        return {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
    
    async def upload_document(
        self,
        file: UploadFile,
        api_key: str,
        collection_name: str,
        wait_for_processing: bool = True
    ) -> Dict:
        """
        Upload a single document to ColiVara
        
        Args:
            file: Uploaded file
            api_key: API key for authentication
            collection_name: Collection to upload to
            wait_for_processing: Whether to wait for processing
            
        Returns:
            Upload response dictionary
            
        Raises:
            HTTPException on errors
        """
        try:
            logger.info(f"Starting upload: {file.filename}")
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Check if document already exists
            existing_docs = await self.get_existing_documents(headers, collection_name)
            if file.filename in existing_docs:
                logger.info(f"Document {file.filename} already exists")
                existing_doc = await self.get_document_info(headers, collection_name, file.filename)
                if existing_doc:
                    return {
                        "message": f"Document already exists in collection '{collection_name}'",
                        "filename": existing_doc["name"],
                        "size": existing_doc.get("metadata", {}).get("file_size", 0),
                        "collection_name": collection_name,
                        "document_id": existing_doc["id"],
                        "existing_metadata": existing_doc.get("metadata", {})
                    }
            
            # Read file content
            content = await file.read()
            file_size = len(content)
            file_base64 = base64.b64encode(content).decode('utf-8')
            current_time = datetime.now(timezone(timedelta(hours=8)))
            
            # Enhanced metadata structure
            payload = {
                "name": file.filename,
                "collection_name": collection_name,
                "base64": file_base64,
                "wait": wait_for_processing,
                "metadata": {
                    "file_size": file_size,
                    "content_type": file.content_type,
                    "original_filename": file.filename,
                    "upload_timestamp": current_time.isoformat(),
                    "processing_time": current_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "upload_method": "single",
                    "source_system": self.get_system_info(),
                    "upload_parameters": {
                        "collection": collection_name,
                        "wait": wait_for_processing
                    },
                    "file_characteristics": {
                        "extension": os.path.splitext(file.filename)[1]
                    }
                }
            }
            
            # Upload to ColiVara
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/documents/upsert-document/",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 401:
                    raise HTTPException(status_code=401, detail="Invalid API key")
                elif response.status_code == 413:
                    raise HTTPException(status_code=413, detail="File too large for server")
                elif response.status_code not in [200, 201]:
                    error_detail = response.text if response.text else f"HTTP {response.status_code}"
                    logger.error(f"API error: {error_detail}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"ColiVara API error: {error_detail}"
                    )
                
                result = response.json()
                return {
                    "message": "Document uploaded successfully",
                    "filename": file.filename,
                    "size": file_size,
                    "collection_name": collection_name,
                    "document_id": result.get("id", file.filename)
                }
        
        except HTTPException:
            raise
        except httpx.TimeoutException:
            logger.error(f"Upload timeout for file: {file.filename}")
            raise HTTPException(status_code=504, detail="Upload timeout - file may be too large")
        except Exception as e:
            logger.exception(f"Upload error for {file.filename}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    async def upload_single_file(
        self,
        semaphore: asyncio.Semaphore,
        headers: dict,
        file_path: str,
        filename: str,
        collection_name: str,
        continue_on_error: bool
    ) -> bool:
        """
        Upload a single file from file system
        
        Args:
            semaphore: Concurrency control
            headers: Authentication headers
            file_path: Path to file
            filename: File name
            collection_name: Collection name
            continue_on_error: Whether to continue on error
            
        Returns:
            True if successful, False otherwise
        """
        async with semaphore:
            try:
                # Read file content
                async with aiofiles.open(file_path, "rb") as f:
                    content = await f.read()
                
                file_size = len(content)
                file_base64 = base64.b64encode(content).decode('utf-8')
                current_time = datetime.now(timezone(timedelta(hours=8)))
                file_extension = os.path.splitext(filename)[1]
                
                # Guess content type
                content_type, _ = mimetypes.guess_type(filename)
                if not content_type:
                    content_type = "application/octet-stream"
                
                # Create payload
                payload = {
                    "name": filename,
                    "collection_name": collection_name,
                    "base64": file_base64,
                    "wait": True,
                    "metadata": {
                        "file_size": file_size,
                        "content_type": content_type,
                        "original_filename": filename,
                        "upload_timestamp": current_time.isoformat(),
                        "processing_time": current_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "upload_method": "bulk",
                        "source_system": self.get_system_info(),
                        "file_characteristics": {
                            "extension": file_extension
                        }
                    }
                }
                
                # Upload
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        f"{self.base_url}/documents/upsert-document/",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code in [200, 201]:
                        logger.info(f"Successfully uploaded: {filename}")
                        return True
                    else:
                        logger.error(f"Upload failed for {filename}: {response.text}")
                        if not continue_on_error:
                            raise Exception(f"Upload failed: {response.text}")
                        return False
            
            except Exception as e:
                logger.error(f"Error uploading {filename}: {str(e)}")
                if not continue_on_error:
                    raise
                return False
    
    async def bulk_upload(
        self,
        folder_path: str,
        api_key: str,
        collection_name: str,
        max_file_size_mb: int = 50,
        continue_on_error: bool = True,
        max_concurrent: int = 3
    ) -> Dict:
        """
        Bulk upload documents from a folder
        
        Args:
            folder_path: Path to folder containing documents
            api_key: API key for authentication
            collection_name: Collection to upload to
            max_file_size_mb: Maximum file size in MB
            continue_on_error: Whether to continue on individual file errors
            max_concurrent: Maximum concurrent uploads
            
        Returns:
            Upload results dictionary
        """
        try:
            # Validate folder
            if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
                raise HTTPException(status_code=400, detail="Invalid folder path")
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Get existing documents
            existing_docs = await self.get_existing_documents(headers, collection_name)
            logger.info(f"Found {len(existing_docs)} existing documents")
            
            # Process files
            new_files = []
            max_size_bytes = max_file_size_mb * 1024 * 1024
            skipped_files = {"existing": [], "too_large": [], "invalid": []}
            
            # Supported file extensions
            supported_extensions = {
                '.pdf', '.doc', '.docx', '.txt', '.png', '.jpg',
                '.jpeg', '.tiff', '.bmp', '.csv', '.xls', '.xlsx',
                '.ppt', '.pptx', '.md', '.html', '.json'
            }
            
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                
                if not os.path.isfile(file_path):
                    continue
                
                # Check file extension
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in supported_extensions:
                    skipped_files["invalid"].append(filename)
                    continue
                
                # Check if already exists
                if filename in existing_docs:
                    skipped_files["existing"].append(filename)
                    continue
                
                # Check file size
                file_size = os.path.getsize(file_path)
                if file_size > max_size_bytes:
                    skipped_files["too_large"].append(filename)
                    continue
                
                new_files.append((file_path, filename))
            
            if not new_files:
                return {
                    "message": "No new files to upload",
                    "total_files": 0,
                    "success_count": 0,
                    "failed_files": [],
                    "skipped_files": skipped_files
                }
            
            logger.info(f"Found {len(new_files)} new files to upload")
            
            # Upload files with concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            upload_tasks = []
            
            for file_path, filename in new_files:
                task = self.upload_single_file(
                    semaphore, headers, file_path, filename,
                    collection_name, continue_on_error
                )
                upload_tasks.append(task)
            
            # Wait for all uploads
            results = await asyncio.gather(*upload_tasks, return_exceptions=True)
            
            # Process results
            success_count = 0
            failed_files = []
            
            for i, result in enumerate(results):
                filename = new_files[i][1]
                if isinstance(result, Exception):
                    error_msg = str(result)[:200]
                    failed_files.append({"filename": filename, "error": error_msg})
                elif result:
                    success_count += 1
                else:
                    failed_files.append({"filename": filename, "error": "Upload failed"})
            
            return {
                "message": "Bulk upload complete",
                "total_files": len(new_files),
                "success_count": success_count,
                "failed_files": failed_files,
                "skipped_files": skipped_files
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Bulk upload error")
            raise HTTPException(status_code=500, detail=f"Bulk upload failed: {str(e)}")
    
    async def delete_document(
        self,
        filename: str,
        api_key: str,
        collection_name: str
    ) -> Dict:
        """
        Delete a single document
        
        Args:
            filename: Document filename
            api_key: API key for authentication
            collection_name: Collection name
            
        Returns:
            Delete response dictionary
        """
        try:
            decoded_filename = urllib.parse.unquote(filename)
            logger.info(f"Deleting document: '{decoded_filename}'")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # URL-safe encoding
            encoded_filename = urllib.parse.quote(decoded_filename, safe='')
            
            async with httpx.AsyncClient() as client:
                delete_response = await client.delete(
                    f"{self.base_url}/documents/delete-document/{encoded_filename}/",
                    headers=headers,
                    params={"collection_name": collection_name}
                )
                
                if delete_response.status_code == 204:
                    return {
                        "message": f"Document '{decoded_filename}' deleted successfully",
                        "filename": decoded_filename,
                        "collection_name": collection_name
                    }
                elif delete_response.status_code == 404:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Document '{decoded_filename}' not found"
                    )
                else:
                    raise HTTPException(
                        status_code=delete_response.status_code,
                        detail=f"Delete failed: {delete_response.text}"
                    )
        
        except HTTPException:
            raise
        except httpx.RequestError as e:
            logger.error(f"Network error deleting document: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
    
    async def delete_matching_documents(
        self,
        keyword: str,
        api_key: str,
        collection_name: str
    ) -> Dict:
        """
        Delete all documents matching a keyword
        
        Args:
            keyword: Keyword to match in document names
            api_key: API key for authentication
            collection_name: Collection name
            
        Returns:
            Delete results dictionary
        """
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Get all documents
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/documents/",
                    headers=headers,
                    params={"collection_name": collection_name}
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Failed to list documents: {response.text}"
                    )
                
                documents = response.json()
            
            # Find matching documents
            keyword_lower = keyword.lower()
            matching_docs = [
                doc for doc in documents
                if keyword_lower in doc.get("name", "").lower()
            ]
            
            if not matching_docs:
                return {
                    "message": f"No documents found matching '{keyword}'",
                    "matched_count": 0,
                    "deleted_count": 0,
                    "deleted": [],
                    "failed": []
                }
            
            # Delete matching documents
            deleted = []
            failed = []
            
            async with httpx.AsyncClient() as client:
                for doc in matching_docs:
                    filename = doc.get("name", "")
                    encoded_filename = urllib.parse.quote(filename, safe='')
                    delete_url = f"{self.base_url}/documents/delete-document/{encoded_filename}/"
                    
                    delete_resp = await client.delete(
                        delete_url,
                        headers=headers,
                        params={"collection_name": collection_name}
                    )
                    
                    if delete_resp.status_code == 204:
                        deleted.append(filename)
                    else:
                        failed.append({"filename": filename, "error": delete_resp.text})
            
            return {
                "message": f"Bulk delete completed for keyword '{keyword}'",
                "matched_count": len(matching_docs),
                "deleted_count": len(deleted),
                "deleted": deleted,
                "failed": failed
            }
        
        except Exception as e:
            logger.exception("Bulk delete error")
            raise HTTPException(status_code=500, detail=f"Bulk delete failed: {str(e)}")
    
    async def list_documents(
        self,
        api_key: str,
        collection_name: str
    ) -> Dict:
        """
        List all documents in a collection
        
        Args:
            api_key: API key for authentication
            collection_name: Collection name
            
        Returns:
            List of documents
        """
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/documents/",
                    headers=headers,
                    params={"collection_name": collection_name}
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Failed to list documents: {response.text}"
                    )
                
                documents = response.json()
                
                # Format documents
                document_list = []
                for doc in documents:
                    document_list.append({
                        "name": doc.get("name", ""),
                        "id": doc.get("id"),
                        "size": doc.get("metadata", {}).get("file_size"),
                        "collection": collection_name,
                        "metadata": doc.get("metadata", {})
                    })
                
                return {
                    "documents": document_list,
                    "total_count": len(document_list),
                    "collection_name": collection_name
                }
        
        except Exception as e:
            logger.exception("List documents error")
            raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
