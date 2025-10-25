"""Processors package for data processing"""
from .table_processor import TableProcessor, TableDetectionConfig, table_processor

"""
Data Processors Package
"""

from .table_processor import TableProcessor
from .query_processor import QueryProcessor

__all__ = [
    'TableProcessor',
    'QueryProcessor',
]
