"""
Pydantic schemas for API requests and responses.
Defines data models for validation and serialization.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union


class QueryRequest(BaseModel):
    """Request model for document queries"""
    api_key: str = Field(..., description="ColiVara API key for authentication")
    query: str = Field(..., description="Search query")
    include_next_page: bool = Field(default=False, description="Include next consecutive pages")
    max_additional_pages: int = Field(default=1, description="Maximum additional pages to fetch", ge=0, le=10)
    allow_general_fallback: bool = Field(default=True, description="Allow general knowledge when documents not found")
    similarity_threshold: float = Field(default=0.25, description="Minimum match confidence (0-1)", ge=0.0, le=1.0)
    collection_name: str = Field(default="default_collection", description="Collection to search in")


class ImageInfo(BaseModel):
    """Information about an image in the response"""
    document_name: str
    page_number: int
    base64_preview: Optional[str] = None
    size: Optional[int] = None


class QueryResponse(BaseModel):
    """Response model for document queries"""
    result: str = Field(..., description="Query result text")
    table_data: Optional[Dict] = Field(None, description="Extracted table data if present")
    source_info: Dict = Field(..., description="Information about document sources")
    debug_info: List[str] = Field(default_factory=list, description="Debug information")
    images_info: List[ImageInfo] = Field(default_factory=list, description="Information about source images")
    base64_preview: Optional[str] = Field(None, description="Preview of base64 image data")
    diagram_base64: Optional[str] = Field(None, description="Base64 encoded diagram image")
    mermaid_syntax: Optional[str] = Field(None, description="Mermaid diagram syntax")
    full_response: Optional[Dict] = Field(None, description="Full response from the model")


class UploadResponse(BaseModel):
    """Response model for file uploads"""
    message: str
    filename: str
    size: int
    collection_name: str = "default_collection"
    document_id: Optional[Union[str, int]] = None
    pages: Optional[int] = None


class BulkUploadResponse(BaseModel):
    """Response model for bulk file uploads"""
    message: str
    successful: List[Dict]
    failed: List[Dict]
    total: int
    successful_count: int
    failed_count: int


class DeleteResponse(BaseModel):
    """Response model for file deletions"""
    message: str
    filename: str
    collection_name: Optional[str] = None


class DeleteMatchingResponse(BaseModel):
    """Response model for pattern-based deletions"""
    message: str
    deleted_count: int
    deleted_documents: List[str]
    collection_name: str


class DocumentInfo(BaseModel):
    """Information about a document"""
    name: str
    size: Optional[int] = None
    last_modified: Optional[str] = None
    collection: Optional[str] = None
    pages: Optional[int] = None


class DocumentListResponse(BaseModel):
    """Response model for document listing"""
    documents: List[DocumentInfo]
    total_count: int
    collection_name: str


class DocumentSearchResponse(BaseModel):
    """Response model for document search"""
    documents: List[DocumentInfo]
    total_count: int
    search_term: str


class PagePreview(BaseModel):
    """Preview of a document page"""
    page_number: int
    base64_image: str
    document_name: str


class DocumentPreviewResponse(BaseModel):
    """Response model for document preview"""
    document_name: str
    pages: List[PagePreview]
    total_pages: int
    collection_name: str


class DocumentExistsResponse(BaseModel):
    """Response model for document existence check"""
    exists: bool
    document_name: str
    collection_name: str
    message: str


class UploadStatusResponse(BaseModel):
    """Response model for upload status check"""
    status: str  # 'pending', 'processing', 'completed', 'failed'
    document_name: str
    progress: Optional[float] = None  # 0.0 to 1.0
    message: Optional[str] = None
    uploaded_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    services: Optional[Dict[str, str]] = None
    version: str = "2.0.0"


class SystemInfoResponse(BaseModel):
    """Response model for system information"""
    message: str
    version: str
    endpoints: Dict[str, str]
    authentication: Dict[str, Union[List[str], str]]
    configurations: Dict[str, str]
    supported_file_types: List[str]
    examples: Dict[str, str]
