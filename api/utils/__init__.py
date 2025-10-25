"""
API Utilities Package

Contains utility modules for:
- Image processing and validation
- Text processing and normalization  
- Input validation and quality checks
- General helper functions
"""

from .image_utils import validate_base64_image, combine_images_for_vlm
from .validation import validate_response_quality, is_relevant_section
from .text_processor import (
    detect_query_language,
    preprocess_query,
    expand_query_terms,
    convert_chinese_to_arabic,
    convert_arabic_to_chinese_numeral,
    normalize_legal_references,
    normalize_numbers_fast,
    normalize_mathematical_expressions,
    extract_mathematical_expressions,
    perform_basic_calculations,
    get_section_context,
    preserve_original_structure
)

__all__ = [
    # Image utilities
    'validate_base64_image',
    'combine_images_for_vlm',
    
    # Validation utilities
    'validate_response_quality',
    'is_relevant_section',
    
    # Text processing utilities
    'detect_query_language',
    'preprocess_query',
    'expand_query_terms',
    'convert_chinese_to_arabic',
    'convert_arabic_to_chinese_numeral',
    'normalize_legal_references',
    'normalize_numbers_fast',
    'normalize_mathematical_expressions',
    'extract_mathematical_expressions',
    'perform_basic_calculations',
    'get_section_context',
    'preserve_original_structure'
]
