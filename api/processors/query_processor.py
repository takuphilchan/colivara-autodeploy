"""
Query Processor - Simplified extraction from original api.py
Handles query expansion, text processing, and relevance scoring
"""

import re
import logging
from typing import List, Dict, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


# Pre-compiled patterns for performance
NUMBERED_LIST_PATTERN = re.compile(r'(\d+)\.\s+')
BULLET_PATTERN = re.compile(r'([•·])\s+')
SECTION_HEADER_PATTERN = re.compile(r'第([一二三四五六七八九十百千万]+)([章条款项节])')
WHITESPACE_PATTERN = re.compile(r'\s+')
MULTIPLE_NEWLINES_PATTERN = re.compile(r'\n{3,}')


class QueryProcessor:
    """Simplified query processor for text analysis and relevance scoring"""
    
    def __init__(self):
        self.chinese_to_arabic = self._build_chinese_number_map()
    
    def _build_chinese_number_map(self) -> Dict[str, int]:
        """Build Chinese to Arabic number mapping"""
        mapping = {
            '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
            '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
            '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20
        }
        return mapping
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        if not text:
            return ""
        
        # Basic normalization
        text = text.strip()
        text = WHITESPACE_PATTERN.sub(' ', text)
        text = MULTIPLE_NEWLINES_PATTERN.sub('\n\n', text)
        
        return text
    
    def expand_query_terms(self, query: str) -> str:
        """Expand query with relevant terms"""
        # For now, return normalized query
        # Can be enhanced with semantic expansion
        return self.normalize_text(query)
    
    def is_relevant_section(self, text: str, query: str, threshold: float = 0.7) -> bool:
        """Check if text is relevant to query"""
        if not text or not query:
            return False
        
        # Simple relevance check - can be enhanced
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Check for direct match
        if query_lower in text_lower:
            return True
        
        # Check for word overlap
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        
        if not query_words:
            return False
        
        overlap = len(query_words & text_words) / len(query_words)
        return overlap >= threshold
    
    def detect_query_language(self, query: str) -> str:
        """Detect primary language of query"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        return 'chinese' if chinese_chars > english_chars else 'english'
    
    def preserve_structure(self, text: str) -> str:
        """Preserve original text structure"""
        if not text:
            return ""
        
        # Clean up numbered lists
        text = NUMBERED_LIST_PATTERN.sub(r'\1. ', text)
        
        # Clean up bullet points
        text = BULLET_PATTERN.sub(r'\1 ', text)
        
        # Preserve section headers
        text = SECTION_HEADER_PATTERN.sub(r'第\1\2 ', text)
        
        # Clean whitespace
        text = WHITESPACE_PATTERN.sub(' ', text)
        text = MULTIPLE_NEWLINES_PATTERN.sub('\n\n', text)
        
        return text.strip()


# Global instance
query_processor = QueryProcessor()


# Helper functions for compatibility
def preprocess_query(query: str) -> str:
    """Preprocess query text"""
    return query_processor.normalize_text(query)


def expand_query_terms(query: str) -> str:
    """Expand query with relevant terms"""
    return query_processor.expand_query_terms(query)


def is_relevant_section(text: str, query: str) -> bool:
    """Check if text section is relevant to query"""
    return query_processor.is_relevant_section(text, query)


def detect_query_language(query: str) -> str:
    """Detect query language"""
    return query_processor.detect_query_language(query)


def preserve_original_structure(text: str) -> str:
    """Preserve original text structure"""
    return query_processor.preserve_structure(text)
