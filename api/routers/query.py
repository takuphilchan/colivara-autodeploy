"""
Query Router - API endpoint for document querying
Handles the /query endpoint for Q&A over documents
"""

from fastapi import APIRouter, Depends, HTTPException
import logging

from api.services.query_service import QueryService
from api.core.dependencies import verify_api_key
from api.core.config import settings
from api.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter()
service = QueryService()


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents with natural language questions
    
    **Features:**
    - Semantic search across document collection
    - Vision Language Model (VLM) integration
    - Multi-page context understanding
    - Configurable similarity thresholds
    - General knowledge fallback
    
    **Request Parameters:**
    - `query`: Your question in natural language
    - `api_key`: Authentication key for ColiVara
    - `collection_name`: Collection to search (default: "default_collection")
    - `include_next_page`: Include consecutive pages for context
    - `max_additional_pages`: Max additional pages to include (1-10)
    - `allow_general_fallback`: Use general knowledge if no docs found
    - `similarity_threshold`: Minimum match confidence (0.0-1.0)
    
    **Response:**
    - `result`: Answer to your question
    - `source_info`: Information about source documents
    - `debug_info`: Processing steps and debug information
    - `images_info`: Information about document images used
    - `table_data`: Extracted table data (if applicable)
    - `diagram_base64`: Generated diagram (if applicable)
    
    **Example Request:**
    ```json
    {
      "api_key": "your_api_key",
      "query": "What are the requirements for starting a business?",
      "include_next_page": false,
      "max_additional_pages": 1,
      "allow_general_fallback": true,
      "similarity_threshold": 0.25
    }
    ```
    
    **Example Response:**
    ```json
    {
      "result": "Based on the documents...",
      "source_info": {
        "sources": [
          {"document": "business_guide.pdf", "page": 1, "score": 0.95}
        ],
        "total_documents": 1
      },
      "debug_info": ["Starting query", "Found 3 documents"],
      "images_info": []
    }
    ```
    """
    try:
        # Process the query
        result = await service.process_query(
            query=request.query,
            api_key=request.api_key,
            collection_name=request.collection_name,
            include_next_page=request.include_next_page,
            max_additional_pages=request.max_additional_pages,
            allow_general_fallback=request.allow_general_fallback,
            similarity_threshold=request.similarity_threshold
        )
        
        # Return response
        return QueryResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )
