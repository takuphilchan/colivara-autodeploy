"""
Validation Utilities Module

Contains functions for validating:
- Response quality and relevance
- Section relevance to queries
- Input data validation
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def validate_response_quality(response: str, original_query: str, query_processor=None) -> bool:
    """
    Validate that response doesn't hallucinate and is relevant to query
    
    Args:
        response: Generated response text
        original_query: Original user query
        query_processor: Optional ModernQueryProcessor instance for validation
        
    Returns:
        True if response quality is acceptable, False otherwise
    """
    if not response or not original_query:
        return False
    
    # If query processor is available, use its validation
    if query_processor:
        try:
            validation = query_processor.validate_response_relevance(response, original_query)
            return validation.get('overall_relevance', 0) > 0.6
        except Exception as e:
            logger.warning(f"Query processor validation failed: {e}")
    
    # Fallback: basic quality checks
    # Check minimum length
    if len(response.strip()) < 10:
        return False
    
    # Check for repetitive content
    words = response.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # Too repetitive
            return False
    
    # Check if query terms appear in response
    query_terms = set(original_query.lower().split())
    response_lower = response.lower()
    matching_terms = sum(1 for term in query_terms if term in response_lower)
    
    if len(query_terms) > 0:
        match_ratio = matching_terms / len(query_terms)
        return match_ratio > 0.2  # At least some query terms should appear
    
    return True


def is_relevant_section(text: str, query: str, query_processor=None, threshold: float = 0.7) -> bool:
    """
    Check if a text section is relevant to the query using priority-based matching
    
    Args:
        text: Text section to check
        query: Query string
        query_processor: Optional ModernQueryProcessor instance
        threshold: Relevance threshold (0-1)
        
    Returns:
        True if section is relevant, False otherwise
    """
    if not text or not query:
        return False
    
    # Use query processor if available
    if query_processor:
        try:
            relevance_score = query_processor.fuzzy_match_relevance_with_priority(text, query)
            return relevance_score > threshold
        except Exception as e:
            logger.warning(f"Query processor relevance check failed: {e}")
    
    # Fallback: simple keyword matching
    query_terms = set(query.lower().split())
    text_lower = text.lower()
    
    matching_terms = sum(1 for term in query_terms if term in text_lower)
    
    if len(query_terms) > 0:
        match_ratio = matching_terms / len(query_terms)
        return match_ratio > (threshold * 0.8)  # Slightly lower threshold for fallback
    
    return False


def validate_query_input(query: str, min_length: int = 2, max_length: int = 1000) -> tuple[bool, Optional[str]]:
    """
    Validate query input parameters
    
    Args:
        query: Query string to validate
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query:
        return False, "Query cannot be empty"
    
    query_stripped = query.strip()
    
    if len(query_stripped) < min_length:
        return False, f"Query too short (minimum {min_length} characters)"
    
    if len(query_stripped) > max_length:
        return False, f"Query too long (maximum {max_length} characters)"
    
    return True, None


def validate_document_name(name: str) -> tuple[bool, Optional[str]]:
    """
    Validate document name for safety and compatibility
    
    Args:
        name: Document name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Document name cannot be empty"
    
    # Check for malicious patterns
    malicious_patterns = ['..', '/', '\\', '\0', '<', '>', '|', ':', '*', '?', '"']
    for pattern in malicious_patterns:
        if pattern in name:
            return False, f"Document name contains invalid character: {pattern}"
    
    # Check length
    if len(name) > 255:
        return False, "Document name too long (maximum 255 characters)"
    
    return True, None


def validate_base64_format(data: str) -> tuple[bool, Optional[str]]:
    """
    Validate base64 string format (without full decoding)
    
    Args:
        data: Base64 string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not data:
        return False, "Base64 data cannot be empty"
    
    # Remove data URI prefix if present
    if data.startswith('data:'):
        parts = data.split(',', 1)
        if len(parts) != 2:
            return False, "Invalid data URI format"
        data = parts[1]
    
    # Check for valid base64 characters
    import re
    if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', data):
        return False, "Invalid base64 characters"
    
    # Check length is multiple of 4 (after padding)
    if len(data) % 4 != 0:
        return False, "Invalid base64 padding"
    
    return True, None
