from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Tuple
import base64
import urllib.parse
import requests
from colivara_py import ColiVara
import logging
import io
from PIL import Image
import time
import re
from minio import Minio
from minio.error import S3Error
import os
import json
import base64
import uuid
import tempfile
import mimetypes
import platform
from datetime import datetime, timezone, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pydantic import BaseModel
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import logging
import asyncio
import sys
import aiofiles
import jieba
import jieba.posseg as pseg
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional, Union, Any
import math
import logging
from difflib import SequenceMatcher
from functools import lru_cache
import io
import pandas as pd
from tabulate import tabulate
import re
import socket
import unicodedata
from typing import List, Dict, Tuple, Optional, Union, List, Dict, Optional
import logging
from rapidfuzz import fuzz
import jieba
import jieba.posseg as pseg
from dataclasses import dataclass
# MCP Client imports - install with: pip install mcp
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("Warning: MCP not available. Install with: pip install mcp")
    MCP_AVAILABLE = False


security = HTTPBearer()

# Initialize mimetypes
mimetypes.init()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="API for document management and querying with fixed configurations",
    version="1.0.0"
)

# Block common malicious paths
@app.middleware("http")
async def block_malicious_requests(request: Request, call_next):
    MALICIOUS_PATTERNS = [
        r".*/(aspspy|aspxspy)\.(php|jsp|jspx|asp|aspx)",
        r".*(wp-admin|wp-login|xmlrpc\.php).*",
        r".*(\.env|\.git|\.svn).*"
    ]
    
    if any(re.match(pattern, request.url.path) for pattern in MALICIOUS_PATTERNS):
        return JSONResponse(
            status_code=403,
            content={"detail": "Forbidden"},
            headers={"Server": ""}  # Remove server identification
        )
    return await call_next(request)

def get_local_ip():
    """Get the local IP address of the system"""
    try:
        # Connect to a remote server to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        # Fallback to localhost if unable to determine IP
        return "127.0.0.1"

local_ip = get_local_ip()

# Fixed Configuration
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5vl:32b"
MINIO_BUCKET = "colivara"  # Fixed bucket name
DEFAULT_COLLECTION = "default_collection"  # Default collection name
COLIVARA_BASE_URL = f"http://{local_ip}:8001/v1"
COLIVARA_BASE_URL_QUERY = f"http://{local_ip}:8001"
API_KEY = "xhFgoUo3UEdhmlIjtq41ds7QJwDM1Yxo"  # Fixed API key for authentication
MINIO_CONFIG = {
    "endpoint": f"{local_ip}:9000",
    "access_key": "miniokey",
    "secret_key": "miniosecret",
    "secure": False
}

# Initialize MinIO client (for listing/deleting documents if needed)
minio_client = Minio(**MINIO_CONFIG)

# Pydantic models
class QueryRequest(BaseModel):
    api_key: str = Field(default=API_KEY, description="ColiVara API key for authentication")
    query: str = Field(..., description="Search query")
    include_next_page: bool = Field(default=False, description="Include next consecutive pages")
    max_additional_pages: int = Field(default=1, description="Maximum additional pages to fetch")
    allow_general_fallback: bool = Field(default=True, description="Allow general knowledge when documents not found")
    similarity_threshold: float = Field(default=0.25, description="Minimum match confidence (0-1)")

class QueryResponse(BaseModel):
    result: str
    table_data: Optional[Dict] = None
    source_info: Dict
    debug_info: List[str]
    images_info: List[Dict]
    base64_preview: Optional[str] = Field(None, description="First 100 characters of base64 image data")
    diagram_base64: Optional[str] = Field(None, description="Base64 encoded diagram image")
    full_response: Optional[Dict] = Field(None, description="Full response from the model")

class UploadResponse(BaseModel):
    message: str
    filename: str
    size: int
    collection_name: str = DEFAULT_COLLECTION
    document_id: Optional[Union[str, int]] = None

class DeleteResponse(BaseModel):
    message: str
    filename: str

class HealthResponse(BaseModel):
    status: str
    timestamp: float

class TablePatterns:
    def __init__(self):
        self.key_value_pattern = re.compile(r'^\s*[a-zA-Z\u4e00-\u9fff]+\s*[:：]')
        self.space_separators = re.compile(r'\s{2,}')
        self.pipe_separator = re.compile(r'^\s*[\|: \-]+\s*$')
        self.numbering_clean = re.compile(r'^[\d\u2460-\u2468\u2160-\u216F]+[\.\．]?\s*')
        self.bullet_clean = re.compile(r'^[\u2022\u25E6\u25AA\u25AB\u25CF]\s*')
        self.merge_patterns = [
            re.compile(r'^[\-—~]+$'),
            re.compile(r'^\.+$'),
            re.compile(r'^同上$'),
            re.compile(r'^同左$'),
            re.compile(r'^同右$'),
            re.compile(r'^同[上下左右]$'),
        ]
        self.markdown_table = re.compile(r'^\|?.+\|.+\|$')
        self.markdown_separator = re.compile(r'^\|?\s*:?\-{3,}:?\s*(\|?\s*:?\-{3,}:?\s*)*\|?$')
        self.newline_cleanup = re.compile(r'\n{3,}')
        # Additional patterns for post-processing
        self.ai_prefixes = [
            re.compile(r'^(AI|Assistant|根据|Based on|According to).*?[:：]\s*', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^(回答|答案|Answer)[:：]\s*', re.IGNORECASE | re.MULTILINE),
        ]
        self.source_check = re.compile(r'(来源|Source|引用|Citation)', re.IGNORECASE)

_PATTERNS = TablePatterns()

@dataclass
class TableDetectionConfig:
    pipe_ratio: float = 0.5
    colon_ratio: float = 0.6
    space_ratio: float = 0.4
    csv_ratio: float = 0.5
    min_lines: int = 2
    merge_threshold: float = 0.1

class TableProcessor:
    """Modern table processor with optimized detection and parsing"""
    
    def __init__(self, config: Optional[TableDetectionConfig] = None):
        self.config = config or TableDetectionConfig()
        
    @lru_cache(maxsize=256)
    def _cached_detect_table_type(self, lines_hash: int, lines_tuple: tuple) -> str:
        """Cached table type detection to avoid recomputation"""
        return self._detect_table_type_impl(list(lines_tuple))
    
    def detect_table_type(self, lines: List[str]) -> str:
        """Optimized table type detection with better heuristics"""
        lines_tuple = tuple(lines)
        lines_hash = hash(lines_tuple)
        return self._cached_detect_table_type(lines_hash, lines_tuple)
    
    def _detect_table_type_impl(self, lines: List[str]) -> str:
        """
        Improved table type detection with better heuristics - OPTIMIZED
        """
        total_lines = len(lines)
        
        # Early exit for obviously non-table content
        if total_lines < self.config.min_lines:
            return 'none'
        
        # Fast counting with early exits
        pipe_score = 0
        colon_score = 0
        space_score = 0
        csv_score = 0
        
        for line in lines:
            # Pipe check - most common, check first
            pipe_count = line.count('|')
            if pipe_count >= 2:
                pipe_score += 1
                if pipe_count >= 3:
                    pipe_score += 0.5
            
            # Colon check - use compiled regex for key-value detection
            if ':' in line or '：' in line:
                colon_score += 1
                if _PATTERNS.key_value_pattern.match(line.strip()):
                    colon_score += 0.5
            
            # Space check - use compiled regex
            space_matches = _PATTERNS.space_separators.findall(line)
            if space_matches:
                space_score += len(space_matches) * 0.3
            
            # CSV check
            if line.count(',') >= 2 or line.count('\t') >= 1:
                csv_score += 1
        
        # Determine best type with optimized thresholds
        min_threshold = max(2, total_lines * self.config.pipe_ratio)
        
        if pipe_score >= min_threshold:
            return 'pipe'
        elif colon_score >= max(2, total_lines * self.config.colon_ratio):
            return 'colon'
        elif space_score >= max(1.5, total_lines * self.config.space_ratio):
            return 'space'
        elif csv_score >= min_threshold:
            return 'csv'
        
        return 'none'
    
    def is_already_markdown_table(self, text: str) -> bool:
        """Check if text already contains properly formatted markdown tables"""
        lines = text.split('\n')
        markdown_table_lines = 0
        separator_lines = 0
        
        for line in lines:
            line = line.strip()
            if _PATTERNS.markdown_table.match(line):
                markdown_table_lines += 1
            elif _PATTERNS.markdown_separator.match(line):
                separator_lines += 1
        
        # Consider it a markdown table if we have both table lines and separators
        return markdown_table_lines >= 2 and separator_lines >= 1
    
    def process_tables_in_response(self, response_text: str) -> str:
        """Process and convert tables found in response text"""
        try:
            # Split response into potential table blocks
            paragraphs = response_text.split('\n\n')
            processed_paragraphs = []
            
            for paragraph in paragraphs:
                # Check if this paragraph might be a table
                lines = [line.strip() for line in paragraph.split('\n') if line.strip()]
                
                if len(lines) >= 2:
                    table_type = self.detect_table_type(lines)
                    
                    if table_type != 'none':
                        # Try to convert to markdown table
                        markdown_table = self.convert_text_to_markdown_table(paragraph)
                        if markdown_table:
                            processed_paragraphs.append(markdown_table)
                            continue
                
                # Not a table or conversion failed, keep original
                processed_paragraphs.append(paragraph)
            
            return '\n\n'.join(processed_paragraphs)
        except Exception as e:
            logger.warning(f"Table processing failed: {e}")
            return response_text
    
    def parse_table_from_text(self, text: str) -> Optional[Dict]:
        """Parse table structure from text and return metadata"""
        try:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            table_type = self.detect_table_type(lines)
            
            if table_type == 'none':
                return None
            
            # Count table characteristics
            table_lines = []
            for line in lines:
                if '|' in line and line.count('|') >= 2:
                    table_lines.append(line)
            
            if not table_lines:
                return None
            
            # Extract basic table info
            first_row_cells = self.extract_pipe_cells(table_lines[0])
            
            return {
                'format': table_type,
                'rows': len(table_lines),
                'columns': len(first_row_cells) if first_row_cells else 0,
                'has_headers': len(table_lines) > 1
            }
        except Exception as e:
            logger.warning(f"Table parsing failed: {e}")
            return None
    
    def convert_text_to_markdown_table(self, text: str) -> Optional[str]:
        """
        Modern method: Convert any text table to markdown with better structure preservation
        Handles both merged and non-merged tables more reliably
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return None
        
        table_type = self.detect_table_type(lines)
        
        if table_type == 'none':
            return None
        
        # Parse based on detected type with improved logic
        table_data = None
        
        if table_type == 'pipe':
            table_data = self.parse_pipe_table_improved(lines)
        elif table_type == 'colon':
            table_data = self.parse_colon_table_improved(lines)
        elif table_type == 'space':
            table_data = self.parse_space_table_improved(lines)
        elif table_type == 'csv':
            table_data = self.parse_with_pandas_improved(text)
        
        # Early exit if no valid data
        if not table_data or not table_data.get('rows') or not table_data.get('headers'):
            return None
        
        # Convert to markdown using optimized method
        try:
            headers = table_data['headers']
            rows = table_data['rows']
            
            # Normalize row lengths efficiently
            max_cols = len(headers)
            normalized_rows = []
            for row in rows:
                if len(row) > max_cols:
                    row = row[:max_cols]  # Truncate
                elif len(row) < max_cols:
                    row = row + [''] * (max_cols - len(row))  # Pad
                normalized_rows.append(row)
            
            # Use tabulate for clean markdown generation
            return tabulate(
                normalized_rows, 
                headers=headers,
                tablefmt="pipe",
                showindex=False
            )
        except Exception:
            # Fallback to manual creation
            return self.create_markdown_table_manually(headers, normalized_rows)
    
    def parse_pipe_table_improved(self, lines: List[str]) -> Optional[Dict]:
        """
        Improved pipe table parsing with better structure preservation - OPTIMIZED
        """
        # Single-pass filtering with compiled regex
        table_lines = [line for line in lines 
                       if '|' in line and line.count('|') >= 2 
                       and not _PATTERNS.pipe_separator.match(line)]
        
        if not table_lines:
            return None
        
        # Parse all rows efficiently
        all_rows = []
        max_columns = 0
        
        for line in table_lines:
            cells = self.extract_pipe_cells(line)
            if cells:
                all_rows.append(cells)
                max_columns = max(max_columns, len(cells))
        
        if not all_rows:
            return None
        
        # Vectorized normalization
        for row in all_rows:
            row.extend([''] * (max_columns - len(row)))
        
        # Better header detection
        headers, data_rows = self.detect_headers_and_data(all_rows)
        
        # Only apply merge cell logic if we detect merged cells
        if self.detect_merged_cells(all_rows):
            data_rows = self.handle_merged_cells(data_rows, headers)
        
        return {'headers': headers, 'rows': data_rows}
    
    def extract_pipe_cells(self, line: str) -> List[str]:
        """
        Extract cells from a pipe-separated line more carefully - OPTIMIZED
        """
        line = line.strip()
        if not line or '|' not in line:
            return []
        
        # Single split operation with list comprehension
        parts = [part.strip() for part in line.split('|')]
        
        # Remove empty parts from ends efficiently
        while parts and not parts[0]:
            parts.pop(0)
        while parts and not parts[-1]:
            parts.pop(-1)
        
        return parts
    
    def detect_headers_and_data(self, rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        """
        Better logic to detect which row contains headers - OPTIMIZED
        """
        if not rows:
            return [], []
        
        first_row = rows[0]
        
        # Fast heuristic: headers usually contain text, not just numbers
        header_score = sum(1 if re.search(r'[a-zA-Z\u4e00-\u9fff]', cell) else 
                          -1 if re.match(r'^\d+\.?\d*$', cell.strip()) else 0 
                          for cell in first_row if cell.strip())
        
        if header_score > 0 or len(rows) == 1:
            return first_row, rows[1:]
        else:
            # Generate generic headers
            return [f'Column {i+1}' for i in range(len(first_row))], rows
    
    def detect_merged_cells(self, rows: List[List[str]]) -> bool:
        """
        Detect if the table likely contains merged cells - OPTIMIZED
        """
        merge_indicators = 0
        total_cells = 0
        
        for row in rows:
            for cell in row:
                total_cells += 1
                if self.is_merge_indicator_improved(cell.strip()):
                    merge_indicators += 1
        
        # 10% threshold for merge detection
        return total_cells > 0 and (merge_indicators / total_cells) > self.config.merge_threshold
    
    def handle_merged_cells(self, rows: List[List[str]], headers: List[str]) -> List[List[str]]:
        """
        Handle merged cells by carrying forward previous values - OPTIMIZED
        """
        if not rows:
            return rows
        
        processed_rows = []
        
        for row in rows:
            new_row = []
            
            for col_idx, cell in enumerate(row):
                cell = cell.strip()
                
                if self.is_merge_indicator_improved(cell):
                    # Carry forward from previous row
                    if processed_rows and col_idx < len(processed_rows[-1]):
                        new_row.append(processed_rows[-1][col_idx])
                    else:
                        new_row.append('')
                else:
                    new_row.append(cell)
            
            processed_rows.append(new_row)
        
        return processed_rows
    
    def is_merge_indicator_improved(self, cell: str) -> bool:
        """
        More conservative merge indicator detection - OPTIMIZED with compiled patterns
        """
        if not cell:
            return False
        
        # Use pre-compiled patterns for speed
        return any(pattern.match(cell) for pattern in _PATTERNS.merge_patterns)
    
    def parse_colon_table_improved(self, lines: List[str]) -> Optional[Dict]:
        """
        Improved colon-separated table parsing - OPTIMIZED
        """
        rows = []
        
        for line in lines:
            # Check both colon types efficiently
            if '：' in line:
                parts = line.split('：', 1)
            elif ':' in line:
                parts = line.split(':', 1)
            else:
                continue
                
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                
                # Clean up key using compiled patterns
                key = _PATTERNS.numbering_clean.sub('', key)
                key = _PATTERNS.bullet_clean.sub('', key)
                
                if key and value:
                    rows.append([key, value])
        
        return {'headers': ['项目', '内容'], 'rows': rows} if rows else None
    
    def parse_space_table_improved(self, lines: List[str]) -> Optional[Dict]:
        """
        Improved space-separated table parsing with better column detection - OPTIMIZED
        """
        rows = []
        
        # Try simple space splitting first (faster)
        for line in lines:
            cells = _PATTERNS.space_separators.split(line.strip())
            if len(cells) > 1:
                rows.append([cell.strip() for cell in cells])
        
        if not rows:
            return None
        
        # Normalize column count efficiently
        max_cols = max(len(row) for row in rows)
        for row in rows:
            row.extend([''] * (max_cols - len(row)))
        
        # Detect headers
        headers, data_rows = self.detect_headers_and_data(rows)
        
        # Handle merged cells if detected
        if self.detect_merged_cells(rows):
            data_rows = self.handle_merged_cells(data_rows, headers)
        
        return {'headers': headers, 'rows': data_rows}
    
    def parse_with_pandas_improved(self, text: str) -> Optional[Dict]:
        """
        Improved pandas parsing with better separator detection - OPTIMIZED
        """
        # Try most common separators first, limit data size
        for sep in ['\t', ',', ';']:
            try:
                text_io = io.StringIO(text)
                # Use C engine and limit rows for speed
                df = pd.read_csv(text_io, sep=sep, engine='c', nrows=1000, encoding='utf-8')
                
                # Quick validation
                if len(df.columns) > 1 and len(df) > 0:
                    headers = [str(col).strip() for col in df.columns]
                    
                    # Efficient conversion with fillna
                    rows = df.fillna('').astype(str).values.tolist()
                    
                    return {'headers': headers, 'rows': rows}
            except Exception:
                continue
        
        return None
    
    def create_markdown_table_manually(self, headers: List[str], rows: List[List[str]]) -> str:
        """
        Manually create markdown table as fallback - OPTIMIZED
        """
        if not headers or not rows:
            return ""
        
        # Pre-allocate list for efficiency
        result = []
        result.append("| " + " | ".join(headers) + " |")
        result.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Process rows efficiently
        for row in rows:
            # Ensure row has same length as headers
            padded_row = (row + [''] * len(headers))[:len(headers)]
            result.append("| " + " | ".join(padded_row) + " |")
        
        return "\n".join(result)

# Create global instance
table_processor = TableProcessor()

@lru_cache(maxsize=200)
def validate_base64_image(encoded_blob):
    """Validate and clean base64 image data - cached"""
    try:
        if encoded_blob.startswith('data:'):
            encoded_blob = encoded_blob.split(',', 1)[1]
        
        image_bytes = base64.b64decode(encoded_blob)
        image = Image.open(io.BytesIO(image_bytes))
        
        return encoded_blob, len(image_bytes), image.format, image.size
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

# Pre-compiled regex patterns for better performance
NUMBERED_LIST_PATTERN = re.compile(r'^(\s*)(\d+[.\、])\s*', re.MULTILINE)
BULLET_PATTERN = re.compile(r'^(\s*)([•·▪▫])\s*', re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r'[ \t]+')
MULTIPLE_NEWLINES_PATTERN = re.compile(r'\n\s*\n\s*\n+')

# Math patterns
ARITHMETIC_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*([+\-×÷*/])\s*(\d+(?:\.\d+)?)')
PERCENTAGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)%')
FRACTION_PATTERN = re.compile(r'(\d+)/(\d+)')
RANGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*[-至到~]\s*(\d+(?:\.\d+)?)')
CHINESE_MATH_PATTERN = re.compile(r'(加|减|乘|除|等于|大于|小于|不少于|不超过|最多|最少|至少|超过)', re.IGNORECASE)

# Numeric patterns
CHINESE_NUMERIC_PATTERN = re.compile(r'(\d+\.?\d*)([万亿千百])')
RANGE_NORMALIZATION_PATTERN = re.compile(r'(\d+)[\-\~至到](\d+)')
PERCENTAGE_CONVERSION_PATTERN = re.compile(r'百分之(\d+)')
PERCENTAGE_EXPANSION_PATTERN = re.compile(r'(\d+)%')

# Section amplification pattern
SECTION_CLEANUP_PATTERN = re.compile(
    r'(第[零一二三四五六七八九十百千万亿\d]+\s*[条款项目节章部分段篇卷编表图例式附件附录])[：:、．.，,]?\s+([零一二三四五六七八九十百千万亿\d]+)(?=\s|$)'
)

# Enhanced section patterns for multi-layered hierarchies
SECTION_HEADER_PATTERN = re.compile(
    r'^(\s*)(第(?:[一二三四五六七八九十百千万亿]+|\d+)[条款项目节章部分段篇卷编](?:\s*第(?:[一二三四五六七八九十百千万亿]+|\d+)[条款项目节章部分段篇卷编])*)[：:、．.，,]?\s*(.*)',
    re.MULTILINE
)

# Multi-layer hierarchy pattern for nested sections
NESTED_SECTION_PATTERN = re.compile(
    r'(第(?:[一二三四五六七八九十百千万亿]+|\d+)[条款项目节章部分段篇卷编])\s+(第(?:[一二三四五六七八九十百千万亿]+|\d+)[条款项目节章部分段篇卷编])',
    re.UNICODE
)

# Comprehensive hierarchy pattern - handles all levels
HIERARCHY_PATTERN = re.compile(
    r'(第(?:[一二三四五六七八九十百千万亿]+|\d+)[条款项目节章部分段篇卷编])(?:\s*(第(?:[一二三四五六七八九十百千万亿]+|\d+)[条款项目节章部分段篇卷编]))?(?:\s*(第(?:[一二三四五六七八九十百千万亿]+|\d+)[条款项目节章部分段篇卷编]))?(?:\s*(第(?:[一二三四五六七八九十百千万亿]+|\d+)[条款项目节章部分段篇卷编]))?',
    re.UNICODE
)

ENHANCED_SECTION_VALIDATION_PATTERN = re.compile(
    r'第\s*([零一二三四五六七八九十百千万亿]+|\d+)\s*([条款项目节章部分段篇卷编表图例式附件附录])(?:\s+第\s*([零一二三四五六七八九十百千万亿]+|\d+)\s*([条款项目节章部分段篇卷编表图例式附件附录]))?',
    re.UNICODE | re.IGNORECASE
)

# New patterns for fixing output issues
TABLE_CLEANUP_PATTERN = re.compile(r'([—\-]{3,})')
DUPLICATE_HEADER_PATTERN = re.compile(r'(\*\*[^*]+\*\*)\s*\1')
MARKDOWN_CLEANUP_PATTERN = re.compile(r'\*\*([^*]*)\*\*')
REPETITIVE_CONTENT_PATTERN = re.compile(r'(.*?)\1{2,}', re.DOTALL)

# Constants
SECTION_TYPES = ['条', '款', '项', '目', '节', '章', '部分', '段', '篇', '卷', '编', '表', '图', '例', '式', '附件', '附录']
SECTION_TYPES_STR = ''.join(SECTION_TYPES)

# Pre-built mappings for faster lookups
MATH_CONVERSIONS = {
    '加': '+', '减': '-', '乘以': '×', '乘': '×', '除以': '÷', '除': '÷',
    '等于': '=', '大于': '>', '小于': '<', '不少于': '>=', '不超过': '<=',
    '至少': '>=', '最少': '>=', '最多': '<=', '超过': '>', '低于': '<', '高于': '>',
}

# Comprehensive Chinese to Arabic mapping
CHINESE_TO_ARABIC_COMPREHENSIVE = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, 
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '百': 100, '千': 1000, '万': 10000, '亿': 100000000
}

# Arabic to Chinese mapping (for display/normalization)
ARABIC_TO_CHINESE = {
    0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五',
    6: '六', 7: '七', 8: '八', 9: '九', 10: '十'
}

class ModernQueryProcessor:
    """
    Modern query processing with semantic understanding and fuzzy matching.
    Replaces primitive regex-based approaches with SOTA techniques.
    """
    
    def __init__(self):
        # Initialize jieba for Chinese text processing
        jieba.enable_parallel(4)
        
        # Fixed Unicode normalization patterns
        self.unicode_normalizer = {
            'quotes': str.maketrans({
                '\u201c': '"',  # Left double quotation mark
                '\u201d': '"',  # Right double quotation mark  
                '\u2018': "'",  # Left single quotation mark
                '\u2019': "'",  # Right single quotation mark
                '\uff62': '"',  # Halfwidth left corner bracket
                '\uff63': '"',  # Halfwidth right corner bracket
                '\u300c': '"',  # Left corner bracket
                '\u300d': '"',  # Right corner bracket
                '\u300e': '"',  # Left white corner bracket
                '\u300f': '"',  # Right white corner bracket
                '\u301d': '"',  # Reversed double prime quotation mark
                '\u301e': '"'   # Double prime quotation mark
            }),
            'punctuation': str.maketrans({
                '\uff0c': ',',  # Fullwidth comma
                '\u3002': '.',  # Ideographic full stop
                '\uff1b': ';',  # Fullwidth semicolon
                '\uff1a': ':',  # Fullwidth colon
                '\uff1f': '?',  # Fullwidth question mark
                '\uff01': '!',  # Fullwidth exclamation mark
                '\uff08': '(',  # Fullwidth left parenthesis
                '\uff09': ')',  # Fullwidth right parenthesis
                '\u3010': '[',  # Left black lenticular bracket
                '\u3011': ']',  # Right black lenticular bracket
                '\u300a': '<',  # Left double angle bracket
                '\u300b': '>',  # Right double angle bracket
                '\u3001': ',',  # Ideographic comma
                '\u3003': '"',  # Ditto mark
                '\u203b': '*'   # Reference mark
            }),
            'numbers': str.maketrans({
                '\uff10': '0',  # Fullwidth digit zero
                '\uff11': '1',  # Fullwidth digit one
                '\uff12': '2',  # Fullwidth digit two
                '\uff13': '3',  # Fullwidth digit three
                '\uff14': '4',  # Fullwidth digit four
                '\uff15': '5',  # Fullwidth digit five
                '\uff16': '6',  # Fullwidth digit six
                '\uff17': '7',  # Fullwidth digit seven
                '\uff18': '8',  # Fullwidth digit eight
                '\uff19': '9'   # Fullwidth digit nine
            }),
            'spaces': str.maketrans({
                '\u3000': ' ',  # Ideographic space
                '\u00a0': ' ',  # Non-breaking space
                '\u2002': ' ',  # En space
                '\u2003': ' ',  # Em space
                '\u2009': ' '   # Thin space
            })
        }
                        
        # Semantic equivalents for legal/formal documents
        self.semantic_clusters = {
            'requirement_cluster': {
                'core': ['要求', '规定', '必须', '应当', '需要'],
                'variations': ['条件', '资格', '标准', '准则', '指标', '规范', '规章', '制度'],
                'semantic_weight': 0.9
            },
            'entity_cluster': {
                'core': ['企业', '公司', '单位', '机构'],
                'variations': ['组织', '厂商', '厂家', '法人', '商户', '经营者', '团体'],
                'semantic_weight': 0.85
            },
            'action_cluster': {
                'core': ['申请', '办理', '处理', '提交'],
                'variations': ['申报', '提请', '请求', '报名', '递交', '操作', '进行', '完成'],
                'semantic_weight': 0.8
            },
            'calculation_cluster': {
                'core': ['计算', '算', '核算', '统计'],
                'variations': ['测算', '运算', '求', '得出', '估算', '合计'],
                'semantic_weight': 0.95
            },
            'comparison_cluster': {
                'core': ['比较', '对比', '比对'],
                'variations': ['对照', '相比', '区别', '差异', '差别', '异同'],
                'semantic_weight': 0.9
            }
        }
        
        # Document title patterns for better document matching
        self.document_patterns = {
            '工伤保险': ['工伤保险条例', '工伤保险', '工伤条例', '伤残保险'],
            '社会保险': ['社会保险法', '社保法', '社会保险', '社保'],
            '劳动合同': ['劳动合同法', '劳动法', '劳动合同', '用工合同'],
            '公司法': ['公司法', '企业法', '公司条例'],
            '税法': ['税法', '所得税法', '增值税法', '税收'],
            '民法': ['民法典', '民法', '民事法律'],
            '刑法': ['刑法', '刑事法律', '刑罚'],
            '行政法': ['行政法', '行政条例', '行政规定']
        }

        # Add exact match priority configuration
        self.exact_match_priority = {
            'exact_match_boost': 1.0,      # Highest priority
            'normalized_match_boost': 0.95,  # Second highest
            'semantic_match_boost': 0.8,     # Lower priority
            'fuzzy_match_boost': 0.6,        # Lowest priority
            'expansion_threshold': 0.8        # Only expand if query is vague
        }
    
        # Compile fuzzy matching patterns
        self.fuzzy_patterns = self._compile_fuzzy_patterns()
        
    def _compile_fuzzy_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for fuzzy matching with proper Unicode support"""
        return {
            'section_reference': re.compile(
                r'第\s*([零一二三四五六七八九十百千万亿\d]+)\s*([条款项目节章部分段篇卷编表图例式附件附录])(?=\s|[：:、．.，,;；）)]|$)',
                re.UNICODE
            ), 
            'document_reference': re.compile(
                r'([\u4e00-\u9fff]+(?:法|条例|规定|办法|细则|准则|标准|规范|规章|制度))',
                re.UNICODE
            ),
            'mathematical_expression': re.compile(
                r'(\d+(?:\.\d+)?)\s*([+\-×÷*/=<>≤≥])\s*(\d+(?:\.\d+)?)|'
                r'(百分之\d+|[\d.]+%)|'
                r'(\d+(?:\.\d+)?)\s*[-至到~]\s*(\d+(?:\.\d+)?)',
                re.UNICODE
            ),
            'quotation_content': re.compile(
                r'["""''「」『』｢｣〝〞](.*?)["""''「」『』｢｣〝〞]',
                re.UNICODE | re.DOTALL
            ),
            'entity_mentions': re.compile(
                r'(?:公司|企业|机构|单位|组织|个人|自然人)(?:的|之|所|等)?',
                re.UNICODE
            )
        }
    
    @lru_cache(maxsize=1000)
    def normalize_text(self, text: str) -> str:
        """
        Advanced Unicode normalization with proper character handling.
        Replaces primitive regex cleaning.
        """
        if not text:
            return ""
        
        # Step 1: Unicode normalization (NFD -> NFC)
        text = unicodedata.normalize('NFD', text)
        text = unicodedata.normalize('NFC', text)
        
        # Step 2: Apply character mappings
        for mapping in self.unicode_normalizer.values():
            text = text.translate(mapping)
        
        # Step 3: Clean control characters but preserve meaningful whitespace
        text = ''.join(char for char in text 
                      if unicodedata.category(char) not in ['Cc', 'Cf'] 
                      or char in ['\n', '\t', '\r'])
        
        # Step 4: Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    @lru_cache(maxsize=500)
    def extract_semantic_features(self, query: str) -> Dict[str, any]:
        """
        Extract semantic features from query using modern NLP techniques.
        Replaces hardcoded pattern matching.
        """
        normalized_query = self.normalize_text(query)
        
        features = {
            'original_query': query,
            'normalized_query': normalized_query,
            'language': self._detect_language(normalized_query),
            'intent': self._classify_intent(normalized_query),
            'entities': self._extract_entities(normalized_query),
            'semantic_clusters': self._map_to_clusters(normalized_query),
            'quoted_content': self._extract_quoted_content(normalized_query),
            'mathematical_expressions': self._extract_math_expressions(normalized_query),
            'section_references': self._extract_section_refs(normalized_query),
            'document_references': self._extract_document_refs(normalized_query),
            'fuzzy_variants': self._generate_fuzzy_variants(normalized_query),
            'query_specificity': self._calculate_query_specificity(normalized_query)
        }
        
        return features
    
    def _detect_language(self, text: str) -> str:
        """Enhanced language detection"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.findall(r'[\w\u4e00-\u9fff]', text))
        
        if total_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if chinese_ratio > 0.3:
            return 'chinese'
        elif english_ratio > 0.5:
            return 'english'
        else:
            return 'mixed'
    
    def _classify_intent(self, query: str) -> Dict[str, float]:
        """
        Modern intent classification using semantic similarity.
        Replaces primitive keyword matching.
        """
        intent_patterns = {
            'specific_section_request': [
                '第.*条', '第.*款', '第.*项', '第.*节', '第.*章',
                'article', 'section', 'clause', 'paragraph'
            ],
            'document_specific_request': [
                '条例', '法', '规定', '办法', '细则', '准则',
                'regulation', 'law', 'rule', 'provision'
            ],
            'content_request': [
                '内容是什么', '包括哪些', '具体规定', '有哪些要求', '详细介绍',
                'what is', 'what are', 'details', 'content', 'include'
            ],
            'calculation': [
                '计算', '算', '总计', '合计', '求和', '平均', '统计',
                'calculate', 'compute', 'sum', 'total', 'average'
            ],
            'comparison': [
                '比较', '对比', '区别', '不同', '差异', '相比',
                'compare', 'difference', 'versus', 'vs', 'contrast'
            ],
            'explanation': [
                '是什么', '解释', '含义', '意思', '定义', '说明',
                'explain', 'meaning', 'definition', 'what does', 'clarify'
            ],
            'procedure': [
                '如何', '怎样', '步骤', '流程', '方法', '过程',
                'how to', 'process', 'procedure', 'steps', 'method'
            ]
        }
        
        intent_scores = {}
        query_lower = query.lower()
        
        for intent, patterns in intent_patterns.items():
            scores = []
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores.append(1.0)
                else:
                    # Use fuzzy matching for partial matches
                    fuzzy_score = fuzz.partial_ratio(pattern, query_lower) / 100
                    if fuzzy_score > 0.7:
                        scores.append(fuzzy_score)
            
            intent_scores[intent] = max(scores) if scores else 0.0
        
        return intent_scores
    
    def _extract_entities(self, query: str) -> List[Dict]:
        """Extract entities using jieba and fuzzy matching"""
        entities = []
        
        # Chinese entity extraction using jieba
        if self._detect_language(query) in ['chinese', 'mixed']:
            words = pseg.cut(query)
            for word, flag in words:
                if len(word) > 1 and flag in ['n', 'nr', 'ns', 'nt', 'nz']:
                    entities.append({
                        'text': word,
                        'type': 'entity',
                        'pos_tag': flag,
                        'confidence': 0.8
                    })
        
        # Pattern-based entity extraction
        for pattern_name, pattern in self.fuzzy_patterns.items():
            for match in pattern.finditer(query):
                entities.append({
                    'text': match.group(),
                    'type': pattern_name,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        return entities
    
    def _map_to_clusters(self, query: str) -> Dict[str, float]:
        """Map query terms to semantic clusters using fuzzy matching"""
        cluster_scores = {}
        query_lower = query.lower()
        
        for cluster_name, cluster_data in self.semantic_clusters.items():
            max_score = 0.0
            
            # Check core terms (exact match gets full weight)
            for core_term in cluster_data['core']:
                if core_term in query_lower:
                    max_score = cluster_data['semantic_weight']
                    break
            
            # Check variations with fuzzy matching
            if max_score == 0.0:
                for variation in cluster_data['variations']:
                    fuzzy_score = fuzz.partial_ratio(variation, query_lower) / 100
                    if fuzzy_score > 0.8:
                        max_score = max(max_score, fuzzy_score * cluster_data['semantic_weight'] * 0.8)
            
            cluster_scores[cluster_name] = max_score
        
        return cluster_scores
    
    def _extract_quoted_content(self, query: str) -> List[str]:
        """Extract content within various quotation marks"""
        quoted_content = []
        pattern = self.fuzzy_patterns['quotation_content']
        
        for match in pattern.finditer(query):
            content = match.group(1).strip()
            if content:
                quoted_content.append(content)
        
        return quoted_content
    
    def _extract_math_expressions(self, query: str) -> List[Dict]:
        """Extract mathematical expressions with context"""
        expressions = []
        pattern = self.fuzzy_patterns['mathematical_expression']
        
        for match in pattern.finditer(query):
            expressions.append({
                'expression': match.group(),
                'start': match.start(),
                'end': match.end(),
                'type': 'mathematical',
                'normalized': self._normalize_math_expression(match.group())
            })
        
        return expressions
    
    def _normalize_math_expression(self, expr: str) -> str:
        """Normalize mathematical expressions"""
        # Convert Chinese math operators
        math_mapping = {
            '加': '+', '减': '-', '乘': '×', '除': '÷',
            '等于': '=', '大于': '>', '小于': '<',
            '不少于': '>=', '不超过': '<=', '至少': '>=',
            '最多': '<=', '超过': '>', '低于': '<'
        }
        
        normalized = expr
        for chinese, symbol in math_mapping.items():
            normalized = normalized.replace(chinese, symbol)
        
        return normalized
    
    def _extract_section_refs(self, query: str) -> List[Dict]:
        """
        MAJOR FIX: Enhanced section reference extraction that handles ALL variations
        Previous issue: Missing many section reference patterns
        """
        references = []
        
        # ENHANCED: More comprehensive pattern that handles all variations including punctuation
        enhanced_pattern = re.compile(
            r'第\s*([零一二三四五六七八九十百千万亿]+|\d+)\s*([条款项目节章部分段篇卷编表图例式附件附录])(?:\s*[：:、．.，,；;]?\s*第\s*([零一二三四五六七八九十百千万亿]+|\d+)\s*([条款项目节章部分段篇卷编表图例式附件附录]))?(?:\s*[：:、．.，,；;]?\s*第\s*([零一二三四五六七八九十百千万亿]+|\d+)\s*([条款项目节章部分段篇卷编表图例式附件附录]))?(?:\s*[：:、．.，,；;]?\s*第\s*([零一二三四五六七八九十百千万亿]+|\d+)\s*([条款项目节章部分段篇卷编表图例式附件附录]))?',
            re.UNICODE | re.IGNORECASE
        )
        
        for match in enhanced_pattern.finditer(query):
            hierarchy_levels = []
            groups = match.groups()
            
            # Process groups in pairs: (number, type)
            for i in range(0, len(groups), 2):
                if i + 1 < len(groups) and groups[i] and groups[i + 1]:
                    number = groups[i]
                    section_type = groups[i + 1]
                    
                    # ENHANCED: Better conversion with comprehensive error handling
                    try:
                        if number.isdigit():
                            arabic_num = int(number)
                            chinese_num = convert_arabic_to_chinese_numeral(arabic_num)
                            format_type = 'arabic'
                        else:
                            # Validate Chinese numeral characters
                            chinese_chars = set(number)
                            valid_chinese_chars = set('零一二三四五六七八九十百千万亿')
                            
                            if chinese_chars.issubset(valid_chinese_chars):
                                arabic_num = convert_chinese_to_arabic(number)
                                chinese_num = number
                                format_type = 'chinese'
                                
                                # ENHANCED: Better validation for conversion results
                                if arabic_num == 0 and number not in ['零']:
                                    # Try direct lookup in enhanced mappings
                                    manual_lookup = {
                                        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                                        '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17, '十八': 18, '十九': 19,
                                        '二十': 20, '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25,
                                        '三十': 30, '三十一': 31, '三十二': 32, '三十三': 33, '三十四': 34, '三十五': 35,
                                        '四十': 40, '四十一': 41, '四十二': 42, '四十三': 43, '四十四': 44, '四十五': 45,
                                        '五十': 50, '五十一': 51, '五十二': 52, '五十三': 53, '五十四': 54, '五十五': 55
                                    }
                                    if number in manual_lookup:
                                        arabic_num = manual_lookup[number]
                                    else:
                                        continue  # Skip if still can't convert
                            else:
                                # Invalid Chinese numeral, skip
                                continue
                        
                        hierarchy_levels.append({
                            'original_number': number,
                            'arabic_number': str(arabic_num),
                            'chinese_number': chinese_num,
                            'type': section_type,
                            'level': len(hierarchy_levels) + 1,
                            'full_reference': f"第{number}{section_type}",
                            'format': format_type
                        })
                        
                    except Exception as e:
                        print(f"Error processing number '{number}': {e}")
                        # Better fallback handling
                        if number.isdigit():
                            hierarchy_levels.append({
                                'original_number': number,
                                'arabic_number': number,
                                'chinese_number': convert_arabic_to_chinese_numeral(int(number)) if number.isdigit() else number,
                                'type': section_type,
                                'level': len(hierarchy_levels) + 1,
                                'full_reference': f"第{number}{section_type}",
                                'format': 'arabic'
                            })
                        else:
                            # For Chinese numbers that failed conversion, still include them
                            hierarchy_levels.append({
                                'original_number': number,
                                'arabic_number': number,  # Keep as-is
                                'chinese_number': number,
                                'type': section_type,
                                'level': len(hierarchy_levels) + 1,
                                'full_reference': f"第{number}{section_type}",
                                'format': 'chinese'
                            })
            
            if hierarchy_levels:
                # ENHANCED: Generate comprehensive normalized forms with all variations
                normalized_forms = set()
                
                # Add original match
                normalized_forms.add(match.group())
                
                # ENHANCED: Generate all possible search variations
                for level in hierarchy_levels:
                    original_ref = f"第{level['original_number']}{level['type']}"
                    chinese_ref = f"第{level['chinese_number']}{level['type']}"
                    arabic_ref = f"第{level['arabic_number']}{level['type']}"
                    
                    # Basic forms
                    normalized_forms.add(original_ref)
                    normalized_forms.add(chinese_ref)
                    normalized_forms.add(arabic_ref)
                    
                    # Spaced variations
                    normalized_forms.add(f"第 {level['chinese_number']} {level['type']}")
                    normalized_forms.add(f"第 {level['arabic_number']} {level['type']}")
                    normalized_forms.add(f"第{level['chinese_number']} {level['type']}")
                    normalized_forms.add(f"第{level['arabic_number']} {level['type']}")
                    
                    # Punctuation variations
                    for punct in ['：', ':', '，', ',', '。', '.', '；', ';']:
                        normalized_forms.add(f"第{level['chinese_number']}{level['type']}{punct}")
                        normalized_forms.add(f"第{level['arabic_number']}{level['type']}{punct}")
                
                # ENHANCED: Build hierarchy forms with all combinations
                if len(hierarchy_levels) > 1:
                    chinese_parts = [f"第{level['chinese_number']}{level['type']}" for level in hierarchy_levels]
                    arabic_parts = [f"第{level['arabic_number']}{level['type']}" for level in hierarchy_levels]
                    
                    # Various hierarchy combinations
                    normalized_forms.add(' '.join(chinese_parts))
                    normalized_forms.add(' '.join(arabic_parts))
                    normalized_forms.add(''.join(chinese_parts))
                    normalized_forms.add(''.join(arabic_parts))
                    
                    # Mixed combinations
                    for i, level in enumerate(hierarchy_levels):
                        mixed_parts = chinese_parts.copy()
                        mixed_parts[i] = f"第{level['arabic_number']}{level['type']}"
                        normalized_forms.add(' '.join(mixed_parts))
                
                references.append({
                    'full_reference': match.group(),
                    'hierarchy_levels': hierarchy_levels,
                    'normalized_forms': list(normalized_forms),
                    'start': match.start(),
                    'end': match.end(),
                    'normalized': ' '.join([f"第{level['chinese_number']}{level['type']}" for level in hierarchy_levels]),
                    'is_hierarchy': len(hierarchy_levels) > 1,
                    'format': 'mixed'
                })
        
        return references
    
    def _extract_document_refs(self, query: str) -> List[Dict]:
        """Extract document/law references from query"""
        references = []
        pattern = self.fuzzy_patterns['document_reference']
        
        for match in pattern.finditer(query):
            doc_name = match.group(1)
            references.append({
                'document_name': doc_name,
                'start': match.start(),
                'end': match.end(),
                'confidence': self._calculate_document_confidence(doc_name, query)
            })
        
        return references
    
    def _calculate_document_confidence(self, doc_name: str, query: str) -> float:
        """Calculate confidence score for document matching"""
        confidence = 0.7  # Base confidence
        
        # Check if document name appears exactly in query
        if doc_name in query:
            confidence = 0.95
        
        # Check semantic similarity with known document patterns
        for category, patterns in self.document_patterns.items():
            for pattern in patterns:
                similarity = fuzz.ratio(doc_name, pattern) / 100
                if similarity > 0.8:
                    confidence = max(confidence, 0.85 + (similarity - 0.8) * 0.5)
        
        return min(confidence, 1.0)
    
    def _calculate_query_specificity(self, query: str) -> Dict[str, float]:
        """Calculate how specific the query is for different aspects"""
        specificity = {
            'document_specificity': 0.0,
            'section_specificity': 0.0,
            'content_specificity': 0.0,
            'overall_specificity': 0.0
        }
        
        # Document specificity
        doc_pattern = self.fuzzy_patterns['document_reference']
        if doc_pattern.search(query):
            specificity['document_specificity'] = 0.9
        
        # Section specificity
        section_pattern = self.fuzzy_patterns['section_reference']
        if section_pattern.search(query):
            specificity['section_specificity'] = 0.9
        
        # Content specificity (based on query length and complexity)
        words = len(query.split())
        if words >= 5:
            specificity['content_specificity'] = min(0.9, words / 10)
        else:
            specificity['content_specificity'] = words / 5
        
        # Overall specificity
        specificity['overall_specificity'] = (
            specificity['document_specificity'] * 0.4 +
            specificity['section_specificity'] * 0.4 +
            specificity['content_specificity'] * 0.2
        )
        
        return specificity
    
    def _normalize_section_reference(self, ref: str) -> str:
        """Normalize section references - FIXED to properly handle Chinese numerals"""
        if not ref:
            return ref
        
        # Remove extra spaces first
        normalized = re.sub(r'\s+', '', ref)
        
        # Pattern to detect if section already has Chinese numerals
        chinese_section_pattern = re.compile(
            r'第\s*([零一二三四五六七八九十百千万亿]+)\s*([条款项目节章部分段篇卷编表图例式附件附录])(?=\s|[：:、．.，,;；）)]|$)',
            re.UNICODE
        )
        
        # If it already has Chinese numerals, return as-is (don't double-convert)
        if chinese_section_pattern.search(normalized):
            return normalized
        
        # Only convert if it has Arabic numbers
        def replace_arabic_with_chinese(match):
            full_match = match.group(0)
            prefix = match.group(1) if match.group(1) else ""
            arabic_num = match.group(2)
            section_type = match.group(3)
            
            try:
                # Convert Arabic number to Chinese
                num_val = int(arabic_num)
                chinese_num = convert_arabic_to_chinese_numeral(num_val)
                return f"{prefix}第{chinese_num}{section_type}"
            except (ValueError, TypeError):
                # If conversion fails, return original
                return full_match
        
        # Pattern to match section references with Arabic numbers ONLY
        arabic_section_pattern = re.compile(
            r'(第)?\s*(\d+)\s*([条款项目节章部分段篇卷编表图例式附件附录])(?=\s|[：:、．.，,;；）)]|$)',
            re.UNICODE
        )

        normalized = arabic_section_pattern.sub(replace_arabic_with_chinese, normalized)
        
        return normalized

    
    def _generate_fuzzy_variants(self, query: str) -> List[str]:
        """Generate fuzzy variants for better matching"""
        variants = [query]
        
        # Remove common prefixes
        prefixes_to_remove = [
            '请告诉我', '我想知道', '请问', '查询', '搜索', '查找',
            'please tell me', 'i want to know', 'search for'
        ]
        
        for prefix in prefixes_to_remove:
            if query.lower().startswith(prefix.lower()):
                variant = query[len(prefix):].strip()
                if variant:
                    variants.append(variant)
        
        # Remove question marks and exclamations
        cleaned = re.sub(r'[？?！!]+$', '', query).strip()
        if cleaned != query:
            variants.append(cleaned)
        
        # Generate character-level variations for common typos
        if len(query) < 50:  # Only for shorter queries
            variants.extend(self._generate_typo_variants(query))
        
        return list(set(variants))
    
    def _generate_typo_variants(self, text: str) -> List[str]:
        """Generate common typo variants"""
        variants = []
        
        # Common character substitutions in Chinese/English
        substitutions = {
            '的': '得', '得': '的', '在': '再', '再': '在',
            'a': 'e', 'e': 'a', 'o': '0', '0': 'o',
            'i': '1', '1': 'i', 's': '5', '5': 's'
        }
        
        for i, char in enumerate(text):
            if char in substitutions:
                variant = text[:i] + substitutions[char] + text[i+1:]
                variants.append(variant)
        
        return variants
    
    # Additional fix for the query expansion method
    def expand_query_with_semantics(self, query: str) -> Dict[str, List[str]]:
        """
        FIXED: Structured query expansion with exact match priority
        Returns categorized terms instead of a single expanded string
        """
        features = self.extract_semantic_features(query)
        
        # Categorize expansion terms by priority
        expansion_terms = {
            'exact_terms': [],           # Highest priority - exact original terms
            'normalized_terms': [],      # High priority - normalized versions
            'section_variants': [],      # Medium-high priority - section number variants
            'semantic_terms': [],        # Medium priority - semantic equivalents
            'contextual_terms': [],      # Lower priority - contextual additions
            'fuzzy_terms': []           # Lowest priority - fuzzy variants
        }
        
        # 1. HIGHEST PRIORITY: Preserve exact original terms
        original_words = query.split()
        expansion_terms['exact_terms'] = [word.strip() for word in original_words if len(word.strip()) > 1]
        
        # 2. Add normalized query as high priority
        expansion_terms['normalized_terms'].append(features['normalized_query'])
        
        # 3. CRITICAL: Section references with exact preservation
        if features['section_references']:
            for section_ref in features['section_references']:
                # Always keep the original reference as highest priority
                expansion_terms['exact_terms'].append(section_ref['full_reference'])
                
                # Add normalized version as second priority
                if section_ref['normalized'] != section_ref['full_reference']:
                    expansion_terms['section_variants'].append(section_ref['normalized'])
                
                # Only add number variants if they're different from original
                if section_ref.get('is_hierarchy', False):
                    for level in section_ref.get('hierarchy_levels', []):
                        # Add number variants carefully
                        if level.get('format') == 'arabic' and level.get('chinese_number'):
                            chinese_variant = f"第{level['chinese_number']}{level['type']}"
                            if chinese_variant != section_ref['full_reference']:
                                expansion_terms['section_variants'].append(chinese_variant)
                        elif level.get('format') == 'chinese' and level.get('arabic_number'):
                            arabic_variant = f"第{level['arabic_number']}{level['type']}"
                            if arabic_variant != section_ref['full_reference']:
                                expansion_terms['section_variants'].append(arabic_variant)
        
        # 4. Document references - exact preservation
        if features['document_references']:
            for doc_ref in features['document_references']:
                expansion_terms['exact_terms'].append(doc_ref['document_name'])
        
        # 5. Quoted content - exact preservation
        for quoted in features['quoted_content']:
            expansion_terms['exact_terms'].append(quoted)
        
        # 6. CONDITIONAL: Only add semantic terms if query is vague
        query_specificity = features['query_specificity']['overall_specificity']
        if query_specificity < self.exact_match_priority['expansion_threshold']:
            # Add semantic cluster terms sparingly
            for cluster_name, score in features['semantic_clusters'].items():
                if score > 0.8:  # High threshold
                    cluster_data = self.semantic_clusters[cluster_name]
                    # Only add core terms, not variations
                    expansion_terms['semantic_terms'].extend(cluster_data['core'][:1])
        
        # 7. VERY CONDITIONAL: Only add contextual terms for very vague queries
        if query_specificity < 0.5:
            # Add minimal contextual terms
            dominant_intent = max(features['intent'].items(), key=lambda x: x[1])
            if dominant_intent[1] > 0.8:
                intent_map = {
                    'specific_section_request': ['条款'],
                    'document_specific_request': ['法规'],
                    'content_request': ['内容'],
                    'calculation': ['计算'],
                    'comparison': ['比较'],
                    'explanation': ['说明'],
                    'procedure': ['步骤']
                }
                if dominant_intent[0] in intent_map:
                    expansion_terms['contextual_terms'].extend(intent_map[dominant_intent[0]])
        
        # 8. MINIMAL: Only add fuzzy variants for very short/ambiguous queries
        if len(query.split()) <= 3 and query_specificity < 0.3:
            expansion_terms['fuzzy_terms'].extend(features['fuzzy_variants'][:2])
        
        # Remove duplicates within each category
        for category in expansion_terms:
            expansion_terms[category] = list(set(expansion_terms[category]))
        
        return expansion_terms
    
    def validate_section_match(self, text: str, section_ref: Dict) -> Dict:
        """
        MAJOR FIX: Enhanced section validation with position-independent matching
        Previous issue: Position-based penalties were reducing relevance of valid matches
        """
        text_lower = text.lower()
        best_match = {'found': False, 'score': 0.0, 'position': -1, 'matched_form': ''}
        
        # ENHANCED: Check all normalized forms with flexible pattern matching
        for form in section_ref.get('normalized_forms', [section_ref.get('normalized', '')]):
            if not form:
                continue
                
            form_lower = form.lower()
            
            # ENHANCED: Multiple pattern matching strategies
            patterns_to_try = [
                # Exact match with flexible boundaries
                r'(?:^|\s|[：:，,、。．；;])\s*' + re.escape(form_lower) + r'(?:\s*[：:，,、。．；;]|\s|$)',
                # Flexible spacing
                re.escape(form_lower).replace(r'\ ', r'\s*') + r'(?:\s*[：:，,、。．；;]|\s|$)',
                # Just the form with word boundaries
                r'\b' + re.escape(form_lower) + r'\b',
                # Simple substring for fallback
                re.escape(form_lower)
            ]
            
            for i, pattern in enumerate(patterns_to_try):
                try:
                    pattern_match = re.search(pattern, text_lower)
                    if pattern_match:
                        pos = pattern_match.start()
                        # Higher score for more precise patterns
                        score = 1.0 - (i * 0.05)  # Slightly lower score for less precise patterns
                        return {
                            'found': True,
                            'score': score,
                            'position': pos,
                            'matched_form': form,
                            'match_type': f'pattern_{i}',
                            'context': text[max(0, pos-100):pos+len(form)+100]
                        }
                except re.error:
                    continue
        
        # ENHANCED: Individual hierarchy level matching with better coverage
        if section_ref.get('hierarchy_levels'):
            for level_idx, level in enumerate(section_ref['hierarchy_levels']):
                level_patterns = [
                    f"第{level.get('chinese_number', '')}{level.get('type', '')}",
                    f"第{level.get('arabic_number', '')}{level.get('type', '')}",
                    level.get('full_reference', ''),
                    f"第 {level.get('chinese_number', '')} {level.get('type', '')}",
                    f"第 {level.get('arabic_number', '')} {level.get('type', '')}"
                ]
                
                for pattern in level_patterns:
                    if not pattern or pattern == "第":
                        continue
                        
                    pattern_lower = pattern.lower()
                    
                    # ENHANCED: Multiple matching strategies for each pattern
                    search_strategies = [
                        # Strategy 1: Flexible boundary matching
                        r'(?:^|\s|[：:，,、。．；;])\s*' + re.escape(pattern_lower) + r'(?:\s*[：:，,、。．；;]|\s|$)',
                        # Strategy 2: Word boundary matching
                        r'\b' + re.escape(pattern_lower) + r'\b',
                        # Strategy 3: Simple substring
                        re.escape(pattern_lower),
                        # Strategy 4: Fuzzy spacing
                        re.escape(pattern_lower).replace(r'\ ', r'\s*')
                    ]
                    
                    for strategy_idx, search_pattern in enumerate(search_strategies):
                        try:
                            pattern_match = re.search(search_pattern, text_lower)
                            if pattern_match:
                                pos = pattern_match.start()
                                # Score based on strategy precision and level position
                                base_score = 0.95 - (strategy_idx * 0.02)  # Slightly lower for less precise strategies
                                level_penalty = level_idx * 0.01  # Very small penalty for deeper levels
                                score = base_score - level_penalty
                                
                                if score > best_match['score']:
                                    best_match = {
                                        'found': True,
                                        'score': score,
                                        'position': pos,
                                        'matched_form': pattern,
                                        'match_type': f'level_match_strategy_{strategy_idx}',
                                        'context': text[max(0, pos-100):pos+len(pattern)+100]
                                    }
                                    break  # Found a good match, try next pattern
                        except re.error:
                            continue
                    
                    # If we found a high-scoring match, we can return early
                    if best_match['score'] > 0.9:
                        break
        
        # ENHANCED: Partial matching for complex cases
        if not best_match['found'] and section_ref.get('hierarchy_levels'):
            for level in section_ref['hierarchy_levels']:
                number_part = level.get('chinese_number', level.get('arabic_number', ''))
                type_part = level.get('type', '')
                
                if number_part and type_part and len(str(number_part)) > 0:
                    # Look for the number and type separately but close together
                    number_pattern = re.escape(str(number_part).lower())
                    type_pattern = re.escape(type_part.lower())
                    
                    # Pattern that allows some text between number and type
                    combined_pattern = f"第.*?{number_pattern}.*?{type_pattern}"
                    
                    try:
                        partial_match = re.search(combined_pattern, text_lower)
                        if partial_match:
                            pos = partial_match.start()
                            # Lower score for partial matches
                            score = 0.7
                            
                            if score > best_match['score']:
                                best_match = {
                                    'found': True,
                                    'score': score,
                                    'position': pos,
                                    'matched_form': f"第{number_part}{type_part}",
                                    'match_type': 'partial_match',
                                    'context': text[max(0, pos-100):pos+200]
                                }
                    except re.error:
                        continue
        
        return best_match

    def fuzzy_match_relevance_with_priority(self, text: str, query: str, threshold: float = 0.7) -> float:
        """
        MAJOR FIX: Enhanced fuzzy matching with position-independent section scoring
        Previous issue: Position-based penalties reduced relevance of valid section matches
        """
        if not text or not query:
            return 0.0
        
        # Normalize both texts
        norm_text = self.normalize_text(text.lower())
        norm_query = self.normalize_text(query.lower())
        
        # Get structured expansion terms
        expansion_terms = self.expand_query_with_semantics(query)
        
        relevance_scores = []
        
        # ENHANCED: Section-specific validation with position-independent scoring
        features = self.extract_semantic_features(query)
        section_references = features.get('section_references', [])
        
        if section_references:
            section_relevance_scores = []
            multiple_section_matches = 0
            total_sections_requested = len(section_references)
            all_section_matches = []
            
            for section_ref in section_references:
                validation_result = self.validate_section_match(text, section_ref)
                
                if validation_result['found']:
                    multiple_section_matches += 1
                    
                    # ENHANCED: Position-independent scoring
                    base_score = validation_result['score']
                    
                    # Remove position-based penalties - all valid matches get equal treatment
                    position_factor = 1.0  # No position penalty
                    
                    # Score adjustment based on match type
                    if validation_result['match_type'].startswith('pattern_0'):
                        final_score = base_score  # Highest for exact pattern
                    elif validation_result['match_type'].startswith('pattern_1'):
                        final_score = base_score * 0.98  # Very slight reduction
                    elif validation_result['match_type'].startswith('level_match'):
                        final_score = base_score * 0.95  # Slight reduction for level matches
                    elif validation_result['match_type'] == 'partial_match':
                        final_score = base_score * 0.85  # More reduction for partial matches
                    else:
                        final_score = base_score * 0.9  # Default reduction
                    
                    final_score = min(1.0, final_score * position_factor)
                    section_relevance_scores.append(final_score)
                    all_section_matches.append({
                        'score': final_score,
                        'section': section_ref,
                        'validation': validation_result
                    })
            
            if section_relevance_scores:
                # ENHANCED: Better scoring for multiple section matches
                max_section_score = max(section_relevance_scores)
                avg_section_score = sum(section_relevance_scores) / len(section_relevance_scores)
                
                # Coverage bonus for finding multiple requested sections
                coverage_ratio = multiple_section_matches / total_sections_requested
                coverage_boost = coverage_ratio * 0.15  # Increased boost for good coverage
                
                # ENHANCED: Special handling for documents with multiple sections
                if multiple_section_matches > 1:
                    # This document contains multiple requested sections - high priority
                    multi_section_boost = min(0.2, multiple_section_matches * 0.06)
                    
                    # Use weighted average of max and average scores
                    base_score = (max_section_score * 0.7) + (avg_section_score * 0.3)
                    final_score = min(1.0, base_score + coverage_boost + multi_section_boost)
                elif multiple_section_matches == 1:
                    # Single section match - use max score with coverage boost
                    final_score = min(1.0, max_section_score + coverage_boost)
                else:
                    final_score = 0.0
                
                return final_score
        
        # 1. HIGHEST PRIORITY: Exact term matches
        exact_match_scores = []
        for exact_term in expansion_terms['exact_terms']:
            if exact_term.lower() in norm_text:
                # Give different scores based on term importance
                if len(exact_term) > 3:  # Longer terms are more significant
                    exact_match_scores.append(self.exact_match_priority['exact_match_boost'])
                else:
                    exact_match_scores.append(self.exact_match_priority['exact_match_boost'] * 0.9)
        
        if exact_match_scores:
            relevance_scores.append(max(exact_match_scores))
        
        # 2. HIGH PRIORITY: Normalized term matches
        for norm_term in expansion_terms['normalized_terms']:
            if norm_term.lower() in norm_text:
                relevance_scores.append(self.exact_match_priority['normalized_match_boost'])
        
        # 3. MEDIUM-HIGH PRIORITY: Section variant matches
        for section_variant in expansion_terms['section_variants']:
            if section_variant.lower() in norm_text:
                relevance_scores.append(0.9)
        
        # 4. MEDIUM PRIORITY: Semantic term matches
        for semantic_term in expansion_terms['semantic_terms']:
            if semantic_term.lower() in norm_text:
                relevance_scores.append(self.exact_match_priority['semantic_match_boost'])
        
        # 5. LOWER PRIORITY: Contextual term matches
        for contextual_term in expansion_terms['contextual_terms']:
            if contextual_term.lower() in norm_text:
                relevance_scores.append(0.7)
        
        # 6. LOWEST PRIORITY: Fuzzy term matches
        for fuzzy_term in expansion_terms['fuzzy_terms']:
            try:
                from fuzzywuzzy import fuzz
                fuzzy_score = fuzz.partial_ratio(fuzzy_term.lower(), norm_text) / 100
                if fuzzy_score > threshold:
                    relevance_scores.append(fuzzy_score * self.exact_match_priority['fuzzy_match_boost'])
            except ImportError:
                # Simple substring matching fallback
                if fuzzy_term.lower() in norm_text:
                    relevance_scores.append(0.6)
        
        # 7. FALLBACK: Overall fuzzy matching (very low priority)
        if not relevance_scores:
            try:
                from fuzzywuzzy import fuzz
                overall_fuzzy = fuzz.partial_ratio(norm_query, norm_text) / 100
                if overall_fuzzy > threshold:
                    relevance_scores.append(overall_fuzzy * 0.5)
            except ImportError:
                # Simple substring matching fallback
                if norm_query in norm_text:
                    relevance_scores.append(0.5)

        return max(relevance_scores) if relevance_scores else 0.0

    def create_prioritized_search_query(self, query: str) -> str:
        """
        MAJOR FIX: Create search query that properly handles multiple section references
        Previous issue: Overly complex queries for multiple sections reduced search effectiveness
        """
        expansion_terms = self.expand_query_with_semantics(query)
        
        # Special handling for multiple section queries
        features = self.extract_semantic_features(query)
        section_references = features.get('section_references', [])
        
        if len(section_references) > 1:
            # For multiple section queries, create a simpler, more focused search
            section_terms = []
            for section_ref in section_references[:5]:  # Limit to avoid overly long queries
                # Use the most likely to match form
                if section_ref.get('normalized'):
                    section_terms.append(section_ref['normalized'])
                elif section_ref.get('full_reference'):
                    section_terms.append(section_ref['full_reference'])
            
            # For multiple sections, use OR-style search instead of AND
            # This increases the chance of finding documents that contain ANY of the requested sections
            primary_query = ' '.join(section_terms[:3])  # Use top 3 sections as primary
            
            # Add document type context if available
            doc_refs = features.get('document_references', [])
            if doc_refs:
                doc_name = doc_refs[0].get('document_name', '')
                if doc_name:
                    primary_query = f"{doc_name} {primary_query}"
            
            return primary_query
        
        # For single section or non-section queries, use the original logic
        prioritized_query_parts = []
        
        # Start with exact terms (highest weight)
        if expansion_terms['exact_terms']:
            exact_part = ' '.join(expansion_terms['exact_terms'][:3])
            prioritized_query_parts.append(exact_part)
        
        # Add normalized terms if different from exact
        if expansion_terms['normalized_terms']:
            norm_part = ' '.join(expansion_terms['normalized_terms'][:2])
            if norm_part not in prioritized_query_parts:
                prioritized_query_parts.append(norm_part)
        
        # Add section variants sparingly
        if expansion_terms['section_variants']:
            section_part = ' '.join(expansion_terms['section_variants'][:2])
            prioritized_query_parts.append(section_part)
        
        # Only add semantic terms if query is vague
        if features['query_specificity']['overall_specificity'] < 0.7:
            if expansion_terms['semantic_terms']:
                semantic_part = ' '.join(expansion_terms['semantic_terms'][:1])
                prioritized_query_parts.append(semantic_part)
        
        # Join with spaces, keeping it concise
        final_query = ' '.join(prioritized_query_parts)
        
        # Ensure we don't create overly long queries
        if len(final_query.split()) > 15:
            words = final_query.split()
            final_query = ' '.join(words[:15])
        
        return final_query

    
    def validate_response_relevance(self, response: str, original_query: str) -> Dict[str, float]:
        """
        NEW: Validate that response is relevant to original query
        Helps prevent hallucinations from over-expansion
        """
        features = self.extract_semantic_features(original_query)
        
        validation_scores = {
            'exact_match_coverage': 0.0,
            'section_reference_accuracy': 0.0,
            'document_reference_accuracy': 0.0,
            'semantic_drift_penalty': 0.0,
            'overall_relevance': 0.0
        }
        
        # Check exact match coverage
        exact_matches = 0
        total_exact_terms = 0
        for word in original_query.split():
            if len(word.strip()) > 2:
                total_exact_terms += 1
                if word.lower() in response.lower():
                    exact_matches += 1
        
        if total_exact_terms > 0:
            validation_scores['exact_match_coverage'] = exact_matches / total_exact_terms
        
        # Check section reference accuracy
        if features['section_references']:
            section_matches = 0
            for section_ref in features['section_references']:
                if section_ref['full_reference'] in response:
                    section_matches += 1
            validation_scores['section_reference_accuracy'] = section_matches / len(features['section_references'])
        
        # Check document reference accuracy
        if features['document_references']:
            doc_matches = 0
            for doc_ref in features['document_references']:
                if doc_ref['document_name'] in response:
                    doc_matches += 1
            validation_scores['document_reference_accuracy'] = doc_matches / len(features['document_references'])
        
        # Calculate overall relevance
        weights = {
            'exact_match_coverage': 0.4,
            'section_reference_accuracy': 0.3,
            'document_reference_accuracy': 0.2,
            'semantic_drift_penalty': 0.1
        }
        
        validation_scores['overall_relevance'] = sum(
            validation_scores[key] * weights[key] 
            for key in weights
        )
        
        return validation_scores

# Global instance for easy integration
query_processor = ModernQueryProcessor()

# Updated replacement functions with exact match priority
def preprocess_query(query: str) -> str:
    """FIXED: Preserve exact query structure"""
    features = query_processor.extract_semantic_features(query)
    # Return normalized but don't over-process
    return features['normalized_query']

def expand_query_terms(query: str) -> str:
    """FIXED: Controlled expansion with exact match priority"""
    return query_processor.create_prioritized_search_query(query)

def is_relevant_section(text: str, query: str) -> bool:
    """FIXED: Use priority-based matching"""
    relevance_score = query_processor.fuzzy_match_relevance_with_priority(text, query)
    return relevance_score > 0.7

def validate_response_quality(response: str, original_query: str) -> bool:
    """NEW: Validate response doesn't hallucinate"""
    validation = query_processor.validate_response_relevance(response, original_query)
    return validation['overall_relevance'] > 0.6

def detect_query_type(query: str) -> str:
    """Enhanced replacement for the old detect_query_type function"""
    features = query_processor.extract_semantic_features(query)
    intent_scores = features['intent']
    
    if not intent_scores:
        return 'general'
    
    # Return the intent with the highest confidence
    dominant_intent = max(intent_scores.items(), key=lambda x: x[1])
    return dominant_intent[0] if dominant_intent[1] > 0.5 else 'general'

def preserve_original_structure(text: str) -> str:
    """Enhanced structure preservation with better table formatting - optimized"""
    # Use pre-compiled patterns for better performance
    text = NUMBERED_LIST_PATTERN.sub(r'\1\2 ', text)
    text = BULLET_PATTERN.sub(r'\1\2 ', text)
    text = SECTION_HEADER_PATTERN.sub(r'\1\2 ', text)
    
    # Convert tables to clean markdown (simplified)
    text = table_processor.convert_text_to_markdown_table(text)
    
    # Clean up repetitive table separators
    text = TABLE_CLEANUP_PATTERN.sub('---', text)
    
    # Remove duplicate headers
    text = DUPLICATE_HEADER_PATTERN.sub(r'\1', text)
    
    # Clean up whitespace using pre-compiled patterns
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = MULTIPLE_NEWLINES_PATTERN.sub('\n\n', text)
    
    return text.strip()

@lru_cache(maxsize=500)
def detect_query_language(query: str) -> str:
    """Detect primary language of query - cached"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
    english_chars = len(re.findall(r'[a-zA-Z]', query))
    return 'chinese' if chinese_chars > english_chars else 'english'

def extract_mathematical_expressions(text: str) -> List[Dict]:
    """Extract and identify mathematical expressions from text - optimized"""
    expressions = []
    
    # Use pre-compiled patterns
    for pattern in [ARITHMETIC_PATTERN, PERCENTAGE_PATTERN, FRACTION_PATTERN, RANGE_PATTERN, CHINESE_MATH_PATTERN]:
        for match in pattern.finditer(text):
            expressions.append({
                'match': match.group(),
                'start': match.start(),
                'end': match.end(),
                'type': 'mathematical'
            })
    
    return expressions

def normalize_mathematical_expressions(text: str) -> str:
    """Convert Chinese mathematical terms to standardized forms - optimized"""
    # Use pre-built mapping for faster lookups
    for chinese, symbol in MATH_CONVERSIONS.items():
        text = re.sub(f'\\b{chinese}\\b', symbol, text)
    
    return text

def perform_basic_calculations(expression: str) -> Union[float, str]:
    """Safely evaluate basic mathematical expressions - optimized"""
    try:
        # Clean and normalize the expression
        expression = normalize_mathematical_expressions(expression)
        expression = expression.replace('×', '*').replace('÷', '/')
        
        # Only allow safe mathematical operations (faster check)
        if not all(c in '0123456789+-*/.() ' for c in expression):
            return expression
        
        # Evaluate the expression
        result = eval(expression)
        return round(result, 4) if isinstance(result, float) else result
        
    except Exception:
        return expression

@lru_cache(maxsize=100)
def convert_chinese_to_arabic(chinese_num: str) -> int:
    """
    MAJOR FIX: Enhanced Chinese to Arabic conversion with complete coverage for all section numbers
    Previous issue: Failed to convert many common section numbers (especially 7-11 range)
    """
    if not chinese_num or chinese_num.isdigit():
        return int(chinese_num) if chinese_num.isdigit() else 0
    
    chinese_num = chinese_num.strip()
    
    # COMPLETE mappings for all numbers 0-99 (covers all practical legal section numbers)
    simple_mappings = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, 
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        
        # Complete 11-19 mappings
        '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
        '十六': 16, '十七': 17, '十八': 18, '十九': 19, 
        
        # Complete 20-99 mappings
        '二十': 20, '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25,
        '二十六': 26, '二十七': 27, '二十八': 28, '二十九': 29,
        '三十': 30, '三十一': 31, '三十二': 32, '三十三': 33, '三十四': 34, '三十五': 35,
        '三十六': 36, '三十七': 37, '三十八': 38, '三十九': 39,
        '四十': 40, '四十一': 41, '四十二': 42, '四十三': 43, '四十四': 44, '四十五': 45,
        '四十六': 46, '四十七': 47, '四十八': 48, '四十九': 49,
        '五十': 50, '五十一': 51, '五十二': 52, '五十三': 53, '五十四': 54, '五十五': 55,
        '五十六': 56, '五十七': 57, '五十八': 58, '五十九': 59,
        '六十': 60, '六十一': 61, '六十二': 62, '六十三': 63, '六十四': 64, '六十五': 65,
        '六十六': 66, '六十七': 67, '六十八': 68, '六十九': 69,
        '七十': 70, '七十一': 71, '七十二': 72, '七十三': 73, '七十四': 74, '七十五': 75,
        '七十六': 76, '七十七': 77, '七十八': 78, '七十九': 79,
        '八十': 80, '八十一': 81, '八十二': 82, '八十三': 83, '八十四': 84, '八十五': 85,
        '八十六': 86, '八十七': 87, '八十八': 88, '八十九': 89,
        '九十': 90, '九十一': 91, '九十二': 92, '九十三': 93, '九十四': 94, '九十五': 95,
        '九十六': 96, '九十七': 97, '九十八': 98, '九十九': 99,
        
        # Common hundreds for legal documents
        '一百': 100, '二百': 200, '三百': 300, '四百': 400, '五百': 500,
        '六百': 600, '七百': 700, '八百': 800, '九百': 900, '一千': 1000
    }
    
    # PRIORITY: Check simple mappings first (this fixes the main bug)
    if chinese_num in simple_mappings:
        return simple_mappings[chinese_num]
    
    # ENHANCED: Handle 十X patterns with complete coverage
    if chinese_num.startswith('十'):
        if len(chinese_num) == 1:
            return 10
        remainder = chinese_num[1:]
        
        # Direct lookup for known patterns
        if chinese_num in simple_mappings:
            return simple_mappings[chinese_num]
        
        # Fallback calculation for 十 + single digit
        if len(remainder) == 1 and remainder in simple_mappings and simple_mappings[remainder] < 10:
            return 10 + simple_mappings[remainder]
    
    # ENHANCED: Handle X十Y patterns
    tens_pattern = re.match(r'^([一二三四五六七八九])十([一二三四五六七八九]?)$', chinese_num)
    if tens_pattern:
        tens_digit = simple_mappings.get(tens_pattern.group(1), 0)
        ones_digit = simple_mappings.get(tens_pattern.group(2), 0) if tens_pattern.group(2) else 0
        return tens_digit * 10 + ones_digit
    
        # ENHANCED: Complex number parsing with better validation
    try:
        result = 0
        current = 0
        temp_num = 0
        
        i = 0
        while i < len(chinese_num):
            char = chinese_num[i]
            
            if char in CHINESE_TO_ARABIC_COMPREHENSIVE:
                char_value = CHINESE_TO_ARABIC_COMPREHENSIVE[char]
                
                if char in ['一', '二', '三', '四', '五', '六', '七', '八', '九']:
                    temp_num = char_value
                elif char == '零':
                    temp_num = 0
                elif char == '十':
                    if temp_num == 0:
                        temp_num = 1
                    current += temp_num * 10
                    temp_num = 0
                elif char == '百':
                    if temp_num == 0:
                        temp_num = 1
                    current += temp_num * 100
                    temp_num = 0
                elif char == '千':
                    if temp_num == 0:
                        temp_num = 1
                    current += temp_num * 1000
                    temp_num = 0
                elif char == '万':
                    if temp_num == 0 and current == 0:
                        current = 1
                    result += (current + temp_num) * 10000
                    current = 0
                    temp_num = 0
                elif char == '亿':
                    if temp_num == 0 and current == 0:
                        current = 1
                    result += (current + temp_num) * 100000000
                    current = 0
                    temp_num = 0
            i += 1
        
        result += current + temp_num
        return max(0, result)
        
    except Exception as e:
        print(f"Error in complex parsing for '{chinese_num}': {e}")
        # Enhanced fallback with pattern matching
        digits = ''.join(c for c in chinese_num if c.isdigit())
        if digits:
            return int(digits)
        
        # Last resort: try partial matching
        for key, value in simple_mappings.items():
            if key in chinese_num and len(key) > 1:
                return value
        
        return 0
    
@lru_cache(maxsize=100)
def convert_arabic_to_chinese_numeral(number: int) -> str:
    """
    FIXED: Convert Arabic numerals to Chinese legal numerals with complete coverage
    """
    if number == 0:
        return "零"
    
    if number < 0:
        return "负" + convert_arabic_to_chinese_numeral(-number)
    
    # FIXED: Complete mapping for 1-20 including the problematic 7-11 range
    direct_mappings = {
        1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
        11: "十一", 12: "十二", 13: "十三", 14: "十四", 15: "十五",
        16: "十六", 17: "十七", 18: "十八", 19: "十九", 20: "二十"
    }
    
    if number in direct_mappings:
        return direct_mappings[number]
    
    # Handle larger numbers with the existing logic
    # [Keep the rest of your existing convert_arabic_to_chinese_numeral function]
    def convert_section(num, unit_name=""):
        if num == 0:
            return ""
        if num < 10:
            return direct_mappings.get(num, str(num)) + unit_name
        elif num < 21:  # FIXED: Extended range for better coverage
            return direct_mappings.get(num, f"十{direct_mappings.get(num-10, str(num-10))}") + unit_name
        elif num < 100:
            tens = num // 10
            ones = num % 10
            result = direct_mappings[tens] + "十"
            if ones > 0:
                result += direct_mappings[ones]
            return result + unit_name
        # [Rest of the function remains the same]
        elif num < 1000:
            hundreds = num // 100
            remainder = num % 100
            result = direct_mappings[hundreds] + "百"
            if remainder > 0:
                if remainder < 10:
                    result += "零" + direct_mappings[remainder]
                else:
                    tens = remainder // 10
                    ones = remainder % 10
                    if tens == 1:
                        result += "一十" if remainder >= 10 else "零一"
                    else:
                        result += direct_mappings[tens] + "十"
                    if ones > 0:
                        result += direct_mappings[ones]
            return result + unit_name
        elif num < 10000:
            thousands = num // 1000
            remainder = num % 1000
            result = direct_mappings[thousands] + "千"
            if remainder > 0:
                if remainder < 100:
                    result += "零"
                result += convert_section(remainder)
            return result + unit_name
        else:
            return str(num) + unit_name  # Fallback for very large numbers
    
    # Process the number in sections
    if number < 10000:
        return convert_section(number)
    elif number < 100000000:  # Less than 1 yi (100 million)
        wan_part = number // 10000
        remainder = number % 10000
        result = convert_section(wan_part) + "万"
        if remainder > 0:
            if remainder < 1000:
                result += "零"
            result += convert_section(remainder)
        return result
    else:  # 1 yi or more
        yi_part = number // 100000000
        remainder = number % 100000000
        result = convert_section(yi_part) + "亿"
        if remainder > 0:
            if remainder < 10000000:
                result += "零"
            result += convert_arabic_to_chinese_numeral(remainder)
        return result

def normalize_legal_references(text: str) -> str:
    """Normalize legal references - ONLY convert Arabic to Chinese, preserve Chinese as-is"""
    
    # First, identify all existing Chinese references to preserve them
    chinese_references = set()
    chinese_section_pattern = re.compile(
        r'第\s*([零一二三四五六七八九十百千万亿]+)\s*([条款项目节章部分段篇卷编表图例式附件附录])',
        re.UNICODE
    )
    
    for match in chinese_section_pattern.finditer(text):
        chinese_references.add((match.start(), match.end()))
    
    def arabic_to_chinese_conversion(match):
        """Convert Arabic numerals to Chinese only if not overlapping with existing Chinese"""
        match_start, match_end = match.span()
        
        # Check if this overlaps with any Chinese reference
        for chinese_start, chinese_end in chinese_references:
            if (match_start < chinese_end and match_end > chinese_start):
                return match.group(0)  # Don't convert if overlapping
        
        # Safe to convert Arabic to Chinese
        prefix = match.group(1) if match.group(1) else ""
        section_marker = match.group(2) if match.group(2) else "第"
        num_str = match.group(3)
        section_type = match.group(4)
        
        try:
            if num_str.isdigit():
                num = int(num_str)
                chinese_num = convert_arabic_to_chinese_numeral(num)
                return f"{prefix}{section_marker}{chinese_num}{section_type}"
            else:
                return match.group(0)
        except (ValueError, TypeError):
            return match.group(0)
    
    # Pattern for Arabic section references only
    arabic_section_pattern = re.compile(
        r'(\s*)(第)?\s*(\d+)\s*([条款项目节章部分段篇卷编表图例式附件附录])',
        re.UNICODE
    )
    
    # Convert Arabic references that don't conflict with Chinese ones
    normalized_text = arabic_section_pattern.sub(arabic_to_chinese_conversion, text)
    
    # Handle hierarchy references - ensure proper formatting
    hierarchy_pattern = re.compile(
        r'(第[零一二三四五六七八九十百千万亿\d]+[条款项目节章部分段篇卷编])(\s+)(第[零一二三四五六七八九十百千万亿\d]+[条款项目节章部分段篇卷编])',
        re.UNICODE
    )
    
    # Normalize hierarchy spacing
    normalized_text = hierarchy_pattern.sub(r'\1 \3', normalized_text)
    
    # Clean up section formatting
    normalized_text = SECTION_CLEANUP_PATTERN.sub(r'\1', normalized_text)
    
    return normalized_text

@lru_cache(maxsize=500)
def get_section_context(text: str, position: int, window_size=200) -> str:
    """Get context around a section marker - cached"""
    start = max(0, position - window_size)
    end = min(len(text), position + window_size)
    return text[start:end]

def normalize_numbers_fast(text: str) -> str:
    """Enhanced number normalization - converts Arabic section numbers to Chinese"""
    # Convert Arabic section references to Chinese
    text = normalize_legal_references(text)
    
    # Handle ranges - convert Arabic numbers in ranges to Chinese
    def convert_range_to_chinese(match):
        start_num = match.group(1)
        end_num = match.group(2)
        
        try:
            if start_num.isdigit() and end_num.isdigit():
                start_chinese = convert_arabic_to_chinese_numeral(int(start_num))
                end_chinese = convert_arabic_to_chinese_numeral(int(end_num))
                return f"{start_chinese}到{end_chinese}范围 {start_num}-{end_num}"
            else:
                return match.group(0)  # Return original if not pure Arabic
        except:
            return match.group(0)
    
    # Convert Arabic numeric ranges to include Chinese equivalents
    text = re.sub(r'(\d+)[-~至到](\d+)', convert_range_to_chinese, text)
    
    # Handle percentages - keep both formats
    text = PERCENTAGE_CONVERSION_PATTERN.sub(r'\1%', text)
    text = PERCENTAGE_EXPANSION_PATTERN.sub(r'\1 percent \1% 百分之\1', text)
    
    return text

class MermaidAgent:
    """Enhanced Agent with intelligent diagram detection and generation"""
    
    def __init__(self, mcp_config_path="/home/enine/.mcphost.yml"):
        self.mcp_config_path = mcp_config_path
        self.logger = logging.getLogger('mermaid_agent')
        
        # Simplified and more intelligent detection patterns
        self.detection_patterns = {
            'explicit_diagram_requests': [
                # Direct requests in Chinese
                r'生成.*?图', r'制作.*?图', r'创建.*?图', r'画.*?图', r'绘制.*?图',
                r'mermaid', r'图表', r'图形', r'图示', r'示意图', r'可视化',
                # Direct requests in English  
                r'create.*diagram', r'generate.*diagram', r'make.*chart', 
                r'draw.*diagram', r'visualize', r'mermaid'
            ],
            'structural_content_indicators': [
                # Process/workflow indicators
                r'流程|过程|步骤|程序|方法', r'process|workflow|steps|procedure',
                # Hierarchy/organization indicators  
                r'结构|层次|组织|架构', r'structure|hierarchy|organization|architecture',
                # Relationship indicators
                r'关系|连接|关联|联系', r'relationship|connection|association',
                # Timeline/sequence indicators
                r'时间|顺序|序列|计划', r'timeline|sequence|schedule|plan'
            ]
        }
        
        # Intelligent diagram type detection based on content
        self.diagram_type_patterns = {
            'mindmap': [
                r'思维导图|脑图|概念图', r'mindmap|mind.?map|concept.?map',
                r'中心.*分支|核心.*要点', r'central.*topic|main.*idea'
            ],
            'flowchart': [
                r'流程图|工作流|业务流程', r'flowchart|workflow|business.?process',
                r'步骤|流程|过程|程序', r'steps|process|procedure',
                r'决策|判断|条件', r'decision|condition|if.*then'
            ],
            'sequenceDiagram': [
                r'时序图|序列图|交互图', r'sequence.*diagram|interaction|timeline',
                r'参与者|角色|对象', r'participant|actor|object',
                r'消息|请求|响应|调用', r'message|request|response|call'
            ],
            'gantt': [
                r'甘特图|项目计划|时间表', r'gantt|project.*plan|timeline|schedule',
                r'任务|里程碑|工期', r'task|milestone|duration',
                r'开始.*结束|起止时间', r'start.*end|begin.*finish'
            ],
            'pie': [
                r'饼图|扇形图|比例图', r'pie.*chart|proportion|percentage',
                r'占比|比例|分布|构成', r'ratio|distribution|composition'
            ]
        }
        
        # FIXED: Corrected templates with proper mindmap syntax
        self.fallback_templates = {
            'mindmap': """mindmap
    root((核心概念))
        分支1
        子项目1
        子项目2
        分支2
        子项目3
        子项目4""",
            
            'flowchart': """flowchart TD
        A[开始] --> B{判断条件}
        B -->|是| C[执行操作A]
        B -->|否| D[执行操作B]
        C --> E[结束]
        D --> E""",
            
            'sequenceDiagram': """sequenceDiagram
        participant A as 用户
        participant B as 系统
        A->>B: 发送请求
        B-->>A: 返回响应
        Note over A,B: 交互完成""",
            
            'gantt': """gantt
        title 项目时间表
        dateFormat YYYY-MM-DD
        section 阶段1
            任务1: 2024-01-01, 15d
            任务2: after 任务1, 10d
        section 阶段2
            任务3: after 任务2, 20d""",
            
            'pie': """pie title 数据分布
        "类别A" : 45
        "类别B" : 30
        "类别C" : 25"""
        }

    def should_generate_diagram_with_context(self, query: str, kb_content: str = "") -> bool:
        """Intelligent diagram detection using semantic analysis"""
        combined_text = f"{query} {kb_content}".lower()
        
        # 1. Check for explicit diagram requests (highest confidence)
        for pattern in self.detection_patterns['explicit_diagram_requests']:
            if re.search(pattern, combined_text, re.IGNORECASE):
                self.logger.info(f"Explicit diagram request detected: {pattern}")
                return True
        
        # 2. Analyze content structure and complexity
        structure_score = self._calculate_structure_score(combined_text)
        self.logger.info(f"Structure score: {structure_score}")
        
        if structure_score >= 3:
            self.logger.info("Content structure suggests diagram would be beneficial")
            return True
            
        # 3. Check for specific content patterns that benefit from visualization
        content_indicators = 0
        for pattern in self.detection_patterns['structural_content_indicators']:
            if re.search(pattern, combined_text, re.IGNORECASE):
                content_indicators += 1
        
        if content_indicators >= 2:
            self.logger.info(f"Multiple content indicators found ({content_indicators})")
            return True
            
        return False
    
    def _calculate_structure_score(self, content: str) -> int:
        """Calculate how structured the content is (indicates diagram usefulness)"""
        score = 0
        
        # Lists and numbered items
        if re.search(r'\d+[.、]\s+.*', content):
            score += 2
        if re.search(r'[-*•]\s+.*', content):
            score += 1
            
        # Multiple sections/paragraphs
        lines = content.split('\n')
        if len([line for line in lines if line.strip()]) > 5:
            score += 1
            
        # Hierarchical indicators
        if re.search(r'第\d+[章节条款]|section|chapter|part', content, re.IGNORECASE):
            score += 2
            
        # Process/workflow indicators
        if re.search(r'然后|接下来|首先|其次|最后|then|next|first|second|finally', content, re.IGNORECASE):
            score += 1
            
        # Complex relationships
        if re.search(r'包括|包含|由.*组成|consists?.*of|includes?|contains?', content, re.IGNORECASE):
            score += 1
            
        return score

    def extract_diagram_type(self, query: str, kb_content: str = "") -> str:
        """Smart diagram type detection based on content analysis"""
        content = f"{query} {kb_content}".lower()
        type_scores = {}
        
        # Score each diagram type
        for diagram_type, patterns in self.diagram_type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                score += matches
            type_scores[diagram_type] = score
        
        # Context-based bonuses
        if re.search(r'公司|企业|组织|架构', content):
            type_scores['mindmap'] = type_scores.get('mindmap', 0) + 2
            type_scores['flowchart'] = type_scores.get('flowchart', 0) + 1
            
        if re.search(r'时间|日期|计划|进度', content):
            type_scores['gantt'] = type_scores.get('gantt', 0) + 2
            
        if re.search(r'比例|百分比|占.*%', content):
            type_scores['pie'] = type_scores.get('pie', 0) + 2
        
        # Return best match or intelligent default
        if any(score > 0 for score in type_scores.values()):
            best_type = max(type_scores.items(), key=lambda x: x[1])[0]
            self.logger.info(f"Detected diagram type: {best_type} (score: {type_scores[best_type]})")
            return best_type
        
        # Smart default based on content analysis
        if re.search(r'步骤|流程|过程|方法', content):
            return 'flowchart'
        elif re.search(r'概念|要点|分类', content):
            return 'mindmap'
        else:
            return 'flowchart'  # Most versatile default

    def _create_intelligent_prompt(self, query: str, diagram_type: str, images_data: List[Dict] = None) -> str:
        """Create context-aware prompts that use actual document images for better diagrams"""
        
        # Detect language
        is_chinese = len([c for c in query if '\u4e00' <= c <= '\u9fff']) > len(query) / 4
        
        # Build document source information
        sources_info = ""
        if images_data:
            doc_groups = {}
            for img in images_data:
                doc_name = img.get('document_name', 'Unknown Document')
                page_num = img.get('page_number', 'Unknown')
                if doc_name not in doc_groups:
                    doc_groups[doc_name] = []
                doc_groups[doc_name].append(str(page_num))
            
            sources_list = []
            for doc_name, pages in doc_groups.items():
                page_range = f"第{pages[0]}页" if len(pages) == 1 else f"第{min(pages)}-{max(pages)}页"
                sources_list.append(f"• {doc_name} ({page_range})")
            
            sources_info = f"""
    文档来源信息：
    {chr(10).join(sources_list)}

    重要：请仔细查看提供的文档图片，从中提取实际信息来生成图表。不要使用通用示例。
    """
        
        # Create diagram-specific prompts with STRICT output requirements and image analysis
        if diagram_type == 'mindmap':
            if is_chinese:
                return f"""请仔细查看提供的文档图片，根据图片中的实际内容创建一个思维导图。

    用户请求：{query}

    {sources_info}

    任务要求：
    1. **图片分析**：仔细阅读所有提供的文档图片
    2. **内容提取**：从图片中提取相关的概念、分类、层次结构
    3. **结构构建**：基于图片内容建立思维导图的层次关系
    4. **中心主题**：从文档内容中确定最合适的中心主题
    5. **分支内容**：所有分支和子项必须来自文档图片中的实际信息

    输出格式要求：
    1. 必须以"mindmap"开头
    2. 使用root((中心主题))格式 - 中心主题必须来自文档内容
    3. 用2个空格缩进表示层次关系
    4. 所有分支和子项必须基于文档图片中的实际信息
    5. 不要添加文档中没有的内容
    6. 保持文档信息的准确性和层次结构

    CRITICAL: 图表中的所有内容必须来源于提供的文档图片，不得使用通用知识或假设信息。

    重要：只输出纯粹的Mermaid语法，不要任何解释、说明、来源信息或其他文字。

    输出格式示例：
    mindmap
    root((从文档提取的中心概念))
        文档中的分支1
        文档中的子项1
        文档中的子项2
        文档中的分支2
        文档中的子项3
        文档中的子项4

    现在请分析文档图片并生成基于实际内容的思维导图："""
            else:
                return f"""Please carefully examine the provided document images and create a mindmap based on the actual content shown in the images.

    User request: {query}

    {sources_info}

    Task Requirements:
    1. **Image Analysis**: Carefully read all provided document images
    2. **Content Extraction**: Extract relevant concepts, categories, and hierarchical structures from images
    3. **Structure Building**: Build mindmap hierarchy based on image content
    4. **Central Topic**: Determine the most appropriate central topic from document content
    5. **Branch Content**: All branches and sub-items must come from actual information in document images

    Output Format Requirements:
    1. Must start with "mindmap"
    2. Use root((Central Topic)) format - central topic must come from document content
    3. Use 2-space indentation for hierarchy
    4. All branches and sub-items must be based on actual information from document images
    5. Do not add content not present in the documents
    6. Maintain accuracy and hierarchy of document information

    CRITICAL: All content in the diagram must come from the provided document images, no general knowledge or assumptions allowed.

    IMPORTANT: Output ONLY pure Mermaid syntax. No explanations, notes, source information, or other text.

    Output format example:
    mindmap
    root((Central Concept from Documents))
        Branch1 from Documents
        SubItem1 from Documents
        SubItem2 from Documents
        Branch2 from Documents
        SubItem3 from Documents
        SubItem4 from Documents

    Now analyze the document images and generate a mindmap based on actual content:"""

        elif diagram_type == 'flowchart':
            if is_chinese:
                return f"""请仔细查看提供的文档图片，根据图片中的实际内容创建一个流程图。

    用户请求：{query}

    {sources_info}

    任务要求：
    1. **图片分析**：仔细阅读所有提供的文档图片
    2. **流程识别**：从图片中识别步骤、流程、程序或操作顺序
    3. **逻辑构建**：基于文档内容建立正确的流程逻辑
    4. **步骤提取**：所有流程步骤必须来自文档图片中的实际信息
    5. **决策点识别**：如果文档中有判断或选择，准确表示

    输出格式要求：
    1. 必须以"flowchart TD"开头
    2. 所有流程步骤必须基于文档图片中的实际信息
    3. 流程顺序必须符合文档描述的逻辑
    4. 不要添加文档中没有的步骤或判断条件
    5. 保持文档信息的准确性

    CRITICAL: 流程图中的所有节点和连接必须来源于提供的文档图片，不得使用通用知识或假设信息。

    重要：只输出纯粹的Mermaid语法，不要任何解释、说明、来源信息或其他文字。

    输出格式示例：
    flowchart TD
        A[文档中的开始步骤] --> B{{文档中的判断条件}}
        B -->|是| C[文档中的操作A]
        B -->|否| D[文档中的操作B]
        C --> E[文档中的结束步骤]
        D --> E

    现在请分析文档图片并生成基于实际内容的流程图："""
            else:
                return f"""Please carefully examine the provided document images and create a flowchart based on the actual content shown in the images.

    User request: {query}

    {sources_info}

    Task Requirements:
    1. **Image Analysis**: Carefully read all provided document images
    2. **Process Identification**: Identify steps, processes, procedures, or operation sequences from images
    3. **Logic Building**: Build correct process logic based on document content
    4. **Step Extraction**: All process steps must come from actual information in document images
    5. **Decision Point Identification**: If documents contain judgments or choices, represent them accurately

    Output Format Requirements:
    1. Must start with "flowchart TD"
    2. All process steps must be based on actual information from document images
    3. Process sequence must follow the logic described in documents
    4. Do not add steps or decision conditions not present in documents
    5. Maintain accuracy of document information

    CRITICAL: All nodes and connections in the flowchart must come from the provided document images, no general knowledge or assumptions allowed.

    IMPORTANT: Output ONLY pure Mermaid syntax. No explanations, notes, source information, or other text.

    Output format example:
    flowchart TD
        A[Start Step from Document] --> B{{Decision from Document}}
        B -->|Yes| C[Process A from Document]
        B -->|No| D[Process B from Document]
        C --> E[End Step from Document]
        D --> E

    Now analyze the document images and generate a flowchart based on actual content:"""

        elif diagram_type == 'sequenceDiagram':
            if is_chinese:
                return f"""请仔细查看提供的文档图片，根据图片中的实际内容创建一个时序图。

    用户请求：{query}

    {sources_info}

    任务要求：
    1. **图片分析**：仔细阅读所有提供的文档图片
    2. **参与者识别**：从图片中识别相关的角色、系统或对象
    3. **交互分析**：分析文档中描述的交互、通信或时序关系
    4. **消息提取**：所有消息和交互必须来自文档图片中的实际信息
    5. **时序构建**：基于文档内容建立正确的时间顺序

    输出格式要求：
    1. 必须以"sequenceDiagram"开头
    2. 使用participant定义参与者 - 参与者必须来自文档内容
    3. 所有消息传递必须基于文档图片中描述的实际交互
    4. 不要添加文档中没有的参与者或交互
    5. 保持文档描述的时序准确性

    CRITICAL: 时序图中的所有参与者和交互必须来源于提供的文档图片，不得使用通用知识或假设信息。

    重要：只输出纯粹的Mermaid语法，不要任何解释、说明、来源信息或其他文字。

    输出格式示例：
    sequenceDiagram
        participant A as 文档中的参与者1
        participant B as 文档中的参与者2
        A->>B: 文档中的交互1
        B-->>A: 文档中的响应1

    现在请分析文档图片并生成基于实际内容的时序图："""
            else:
                return f"""Please carefully examine the provided document images and create a sequence diagram based on the actual content shown in the images.

    User request: {query}

    {sources_info}

    Task Requirements:
    1. **Image Analysis**: Carefully read all provided document images
    2. **Participant Identification**: Identify relevant actors, systems, or objects from images
    3. **Interaction Analysis**: Analyze interactions, communications, or sequence relationships described in documents
    4. **Message Extraction**: All messages and interactions must come from actual information in document images
    5. **Sequence Building**: Build correct time sequence based on document content

    Output Format Requirements:
    1. Must start with "sequenceDiagram"
    2. Use participant to define actors - participants must come from document content
    3. All message passing must be based on actual interactions described in document images
    4. Do not add participants or interactions not present in documents
    5. Maintain timing accuracy as described in documents

    CRITICAL: All participants and interactions in the sequence diagram must come from the provided document images, no general knowledge or assumptions allowed.

    IMPORTANT: Output ONLY pure Mermaid syntax. No explanations, notes, source information, or other text.

    Output format example:
    sequenceDiagram
        participant A as Participant1 from Document
        participant B as Participant2 from Document
        A->>B: Interaction1 from Document
        B-->>A: Response1 from Document

    Now analyze the document images and generate a sequence diagram based on actual content:"""

        elif diagram_type == 'gantt':
            if is_chinese:
                return f"""请仔细查看提供的文档图片，根据图片中的实际内容创建一个甘特图。

    用户请求：{query}

    {sources_info}

    任务要求：
    1. **图片分析**：仔细阅读所有提供的文档图片
    2. **任务识别**：从图片中识别任务、阶段、里程碑
    3. **时间分析**：分析文档中的时间安排、期限、顺序
    4. **任务提取**：所有任务和时间安排必须来自文档图片中的实际信息
    5. **依赖关系**：基于文档内容建立任务间的依赖关系

    输出格式要求：
    1. 必须以"gantt"开头
    2. 所有任务和时间安排必须基于文档图片中的实际信息
    3. 时间顺序必须符合文档描述的逻辑
    4. 不要添加文档中没有的任务或时间点
    5. 保持文档时间信息的准确性

    CRITICAL: 甘特图中的所有任务和时间安排必须来源于提供的文档图片，不得使用通用知识或假设信息。

    重要：只输出纯粹的Mermaid语法，不要任何解释、说明、来源信息或其他文字。

    输出格式示例：
    gantt
        title 文档中的项目标题
        dateFormat YYYY-MM-DD
        section 文档中的阶段1
            文档中的任务1: 文档日期, 文档工期
            文档中的任务2: after 任务1, 文档工期

    现在请分析文档图片并生成基于实际内容的甘特图："""
            else:
                return f"""Please carefully examine the provided document images and create a gantt chart based on the actual content shown in the images.

    User request: {query}

    {sources_info}

    Task Requirements:
    1. **Image Analysis**: Carefully read all provided document images
    2. **Task Identification**: Identify tasks, phases, milestones from images
    3. **Time Analysis**: Analyze time arrangements, deadlines, sequences in documents
    4. **Task Extraction**: All tasks and scheduling must come from actual information in document images
    5. **Dependency Relations**: Build task dependencies based on document content

    Output Format Requirements:
    1. Must start with "gantt"
    2. All tasks and scheduling must be based on actual information from document images
    3. Time sequence must follow the logic described in documents
    4. Do not add tasks or time points not present in documents
    5. Maintain accuracy of time information from documents

    CRITICAL: All tasks and scheduling in the gantt chart must come from the provided document images, no general knowledge or assumptions allowed.

    IMPORTANT: Output ONLY pure Mermaid syntax. No explanations, notes, source information, or other text.

    Output format example:
    gantt
        title Project Title from Document
        dateFormat YYYY-MM-DD
        section Phase1 from Document
            Task1 from Document: document-date, document-duration
            Task2 from Document: after Task1, document-duration

    Now analyze the document images and generate a gantt chart based on actual content:"""

        elif diagram_type == 'pie':
            if is_chinese:
                return f"""请仔细查看提供的文档图片，根据图片中的实际数据创建一个饼图。

    用户请求：{query}

    {sources_info}

    任务要求：
    1. **图片分析**：仔细阅读所有提供的文档图片
    2. **数据识别**：从图片中识别数值数据、百分比、比例信息
    3. **分类提取**：识别数据的分类标签和对应数值
    4. **数据验证**：确保提取的数值准确且逻辑合理
    5. **标题确定**：从文档内容中确定合适的图表标题

    输出格式要求：
    1. 必须以"pie"开头
    2. 所有数据标签和数值必须来自文档图片中的实际信息
    3. 数值必须准确反映文档中的数据
    4. 不要添加文档中没有的类别或数值
    5. 保持文档数据的准确性

    CRITICAL: 饼图中的所有标签和数值必须来源于提供的文档图片，不得使用通用知识或假设信息。

    重要：只输出纯粹的Mermaid语法，不要任何解释、说明、来源信息或其他文字。

    输出格式示例：
    pie title 文档中的实际标题
        "文档中的类别1" : 文档中的实际数值1
        "文档中的类别2" : 文档中的实际数值2
        "文档中的类别3" : 文档中的实际数值3

    现在请分析文档图片并生成基于实际数据的饼图："""
            else:
                return f"""Please carefully examine the provided document images and create a pie chart based on the actual data shown in the images.

    User request: {query}

    {sources_info}

    Task Requirements:
    1. **Image Analysis**: Carefully read all provided document images
    2. **Data Identification**: Identify numerical data, percentages, proportion information from images
    3. **Category Extraction**: Identify data category labels and corresponding values
    4. **Data Verification**: Ensure extracted values are accurate and logically reasonable
    5. **Title Determination**: Determine appropriate chart title from document content

    Output Format Requirements:
    1. Must start with "pie"
    2. All data labels and values must come from actual information in document images
    3. Values must accurately reflect data from documents
    4. Do not add categories or values not present in documents
    5. Maintain accuracy of document data

    CRITICAL: All labels and values in the pie chart must come from the provided document images, no general knowledge or assumptions allowed.

    IMPORTANT: Output ONLY pure Mermaid syntax. No explanations, notes, source information, or other text.

    Output format example:
    pie title Actual Title from Document
        "Category1 from Document" : actual-value-1
        "Category2 from Document" : actual-value-2
        "Category3 from Document" : actual-value-3

    Now analyze the document images and generate a pie chart based on actual data:"""

        # Generic fallback with strict requirements
        else:
            template = self.fallback_templates.get(diagram_type, self.fallback_templates['flowchart'])
            if is_chinese:
                return f"""请仔细查看提供的文档图片，根据图片中的实际内容创建{diagram_type}图表。

    用户请求：{query}

    {sources_info}

    参考模板：
    {template}

    任务要求：
    1. **图片分析**：仔细阅读所有提供的文档图片
    2. **内容提取**：从图片中提取与用户查询相关的信息
    3. **结构构建**：基于文档内容构建合适的图表结构
    4. **准确表达**：确保图表准确反映文档内容

    CRITICAL: 图表中的所有内容必须来源于提供的文档图片，不得使用通用知识或假设信息。

    重要：只输出纯粹的Mermaid语法，不要任何解释、说明、来源信息或其他文字。
    现在请分析文档图片并生成基于实际内容的图表："""
            else:
                return f"""Please carefully examine the provided document images and create a {diagram_type} diagram based on the actual content shown in the images.

    User request: {query}

    {sources_info}

    Template reference:
    {template}

    Task Requirements:
    1. **Image Analysis**: Carefully read all provided document images
    2. **Content Extraction**: Extract information relevant to user query from images
    3. **Structure Building**: Build appropriate diagram structure based on document content
    4. **Accurate Representation**: Ensure diagram accurately reflects document content

    CRITICAL: All content in the diagram must come from the provided document images, no general knowledge or assumptions allowed.

    IMPORTANT: Output ONLY pure Mermaid syntax. No explanations, notes, source information, or other text.
    Now analyze the document images and generate a diagram based on actual content:"""


    def generate_standalone_mermaid_syntax(self, query: str, kb_content: str = "", images_data: List[Dict] = None) -> str:
        """Generate Mermaid syntax using VLM with actual document images - FIXED VERSION"""
        
        # Validate that we have either kb_content or images_data
        if not images_data or len(images_data) == 0:
            if not kb_content or len(kb_content.strip()) < 50:
                self.logger.warning("No images_data and insufficient kb_content")
                return ""
        
        if images_data:
            self.logger.info(f"Using {len(images_data)} document images for diagram generation")
        else:
            self.logger.info(f"Using text-only kb_content: {len(kb_content)} characters")
        
        diagram_type = self.extract_diagram_type(query, kb_content)
        
        # Create intelligent prompt that works with images
        prompt = self._create_intelligent_prompt(query, diagram_type, images_data)
        
        try:
            if images_data and len(images_data) > 0:
                # Use VLM with images for better diagram generation
                self.logger.info("Using VLM with document images for diagram generation")
                
                # Import the VLM function from the main module
                from __main__ import call_ollama_vlm_multipage
                
                response_text, _ = call_ollama_vlm_multipage(
                    OLLAMA_URL, MODEL_NAME, prompt, images_data, timeout=120
                )
                
                self.logger.info(f"VLM response length: {len(response_text)} characters")
                
            else:
                # Fallback to text-only generation
                self.logger.info("Using text-only generation")
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": MODEL_NAME,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "top_p": 0.9,
                            "repeat_penalty": 1.1,
                            "presence_penalty": 0.6,
                            "frequency_penalty": 0.6,
                            "num_ctx": 6144,
                            "num_predict": 1000,
                            "num_thread": -1,
                            "num_gpu": -1,
                            "seed": 42,
                            "stop": ["```", "注意:", "说明:", "Note:", "Explanation:", "来源:", "Source:", "根据:", "Based on:"]
                        }
                    },
                    timeout=45
                )
                
                if response.status_code != 200:
                    self.logger.error(f"LLM generation failed: {response.status_code}")
                    return self._create_contextual_fallback(diagram_type, query, kb_content)
                
                response_text = response.json().get('response', '')
            
            # Clean and validate the syntax
            cleaned_syntax = self._clean_and_validate_syntax(response_text, diagram_type)
            
            # Additional cleaning step - ensure absolutely no contamination
            final_syntax = self._ensure_pure_mermaid(cleaned_syntax, diagram_type)
            
            if final_syntax and self._basic_syntax_check(final_syntax, diagram_type):
                self.logger.info(f"Successfully generated {diagram_type} syntax with VLM")
                return final_syntax
            else:
                self.logger.warning("Generated syntax invalid, using enhanced fallback")
                return self._create_contextual_fallback(diagram_type, query, kb_content)
                
        except Exception as e:
            self.logger.error(f"Error in syntax generation: {str(e)}")
            return self._create_contextual_fallback(diagram_type, query, kb_content)
    
    def _clean_and_validate_syntax(self, raw_output: str, expected_type: str) -> str:
        """Clean and validate Mermaid syntax with better logic - FIXED VERSION"""
        if not raw_output:
            return ""
        
        # Remove markdown code blocks completely
        cleaned = re.sub(r'```(?:mermaid)?\s*', '', raw_output, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        cleaned = re.sub(r'```', '', cleaned)  # Remove any remaining backticks
        
        # Remove common AI commentary patterns
        lines = cleaned.split('\n')
        diagram_lines = []
        started = False
        
        # Strict patterns to exclude
        exclude_patterns = [
            r'note:', r'说明', r'解释', r'explanation', r'备注', r'提示',
            r'来源', r'source', r'参考', r'reference', r'based on', r'根据',
            r'文档', r'document', r'页面', r'page', r'以上', r'above',
            r'如下', r'below', r'显示', r'show', r'表示', r'represent',
            r'包含', r'contain', r'包括', r'include', r'具体', r'specific'
        ]
        
        for line in lines:
            line_original = line  # Keep original for indentation
            line = line.strip()
            if not line:
                continue
            
            # Check if this line contains excluded content
            should_exclude = any(re.search(pattern, line.lower()) for pattern in exclude_patterns)
            
            # Start collecting when we see a valid diagram type declaration
            if any(line.lower().startswith(dt) for dt in ['mindmap', 'flowchart', 'graph', 'sequencediagram', 'gantt', 'pie']):
                started = True
                diagram_lines.append(line_original.strip())
            elif started and not should_exclude:
                # FIXED: For mindmaps, preserve original indentation
                if expected_type == 'mindmap':
                    # Keep original line with indentation for mindmaps
                    if self._is_valid_mermaid_line(line_original, expected_type):
                        diagram_lines.append(line_original.rstrip())  # Remove trailing spaces only
                else:
                    # For other diagram types, use cleaned line
                    if self._is_valid_mermaid_line(line_original, expected_type):
                        diagram_lines.append(line_original.strip())
            elif started and should_exclude:
                # Stop when we hit explanatory content
                break
        
        if not diagram_lines:
            return ""
        
        # FIXED: Special post-processing for mindmap to fix common issues
        if expected_type == 'mindmap':
            result_lines = []
            for line in diagram_lines:
                # Skip lines that contain source information or explanations
                if not any(keyword in line.lower() for keyword in ['来源', 'source', '文档', 'document', '说明', 'note', '根据', 'based']):
                    # Fix common mindmap indentation issues
                    stripped = line.strip()
                    if stripped == 'mindmap':
                        result_lines.append(stripped)
                    elif stripped.startswith('root((') and stripped.endswith('))'):
                        result_lines.append(f"  {stripped}")
                    elif stripped and not line.startswith(' '):
                        # Top-level branch
                        result_lines.append(f"    {stripped}")
                    else:
                        # Keep existing indentation or add sub-item indentation
                        if line.startswith('  ') and not line.startswith('    '):
                            # Adjust root indentation
                            result_lines.append(f"  {stripped}")
                        elif line.startswith('    '):
                            # Branch level
                            result_lines.append(line.rstrip())
                        elif line.startswith('      '):
                            # Sub-item level
                            result_lines.append(line.rstrip())
                        else:
                            # Default to sub-item
                            result_lines.append(f"      {stripped}")
            return '\n'.join(result_lines)
        else:
            # Final cleaning for non-mindmap diagrams
            result_lines = []
            for line in diagram_lines:
                if not any(keyword in line.lower() for keyword in ['来源', 'source', '文档', 'document', '说明', 'note', '根据', 'based']):
                    result_lines.append(line)
            return '\n'.join(result_lines)

    def _is_valid_mermaid_line(self, line: str, diagram_type: str) -> bool:
        """Check if a line contains valid Mermaid syntax"""
        line_clean = line.strip()
        
        # Empty lines are OK
        if not line_clean:
            return True
        
        # Common Mermaid syntax patterns by type
        if diagram_type == 'mindmap':
            # FIXED: More permissive mindmap validation
            return (line_clean == 'mindmap' or
                    (line_clean.startswith('root((') and line_clean.endswith('))')) or
                    line.startswith('  ') or  # Any indented content is valid for mindmaps
                    (not line.startswith(' ') and  # Top-level branches without indentation
                    not any(keyword in line_clean.lower() for keyword in 
                            ['来源', 'source', '文档', 'document', '说明', 'note', '根据', 'based', '参考', 'reference'])))
        
        elif diagram_type == 'flowchart':
            # Flowchart lines should contain arrows, node definitions, or be the flowchart declaration
            return (line_clean.startswith('flowchart') or
                    '-->' in line_clean or
                    '->' in line_clean or
                    re.match(r'^[A-Z]\[.+\]', line_clean) or  # Node definitions like A[Start]
                    re.match(r'^[A-Z]\{.+\}', line_clean))   # Decision nodes like B{Decision}
        
        elif diagram_type == 'sequenceDiagram':
            return (line_clean.startswith('sequenceDiagram') or
                    'participant' in line_clean or
                    '->>' in line_clean or
                    '-->' in line_clean or
                    'Note over' in line_clean)
        
        elif diagram_type == 'gantt':
            return (line_clean.startswith('gantt') or
                    'title' in line_clean or
                    'dateFormat' in line_clean or
                    'section' in line_clean or
                    ':' in line_clean)  # Task definitions
        
        elif diagram_type == 'pie':
            return (line_clean.startswith('pie') or
                    'title' in line_clean or
                    '"' in line_clean and ':' in line_clean)  # Pie slice definitions
        
        # If we can't determine, be conservative but allow common patterns
        return not any(keyword in line_clean.lower() for keyword in 
                    ['来源', 'source', '文档', 'document', '说明', 'note', '根据', 'based', '参考', 'reference'])

    def _ensure_pure_mermaid(self, syntax: str, diagram_type: str) -> str:
        """Final cleanup to ensure absolutely pure Mermaid syntax"""
        if not syntax:
            return ""
        
        lines = syntax.split('\n')
        pure_lines = []
        
        # Contamination keywords to absolutely exclude
        contamination_keywords = [
            '来源', 'source', '文档', 'document', '说明', 'note', '根据', 'based',
            '参考', 'reference', '以上', 'above', '如下', 'below', '显示', 'show',
            '表示', 'represent', '包含', 'contain', '包括', 'include', '具体', 'specific',
            '页面', 'page', '第', '章', '节', 'chapter', 'section', '内容', 'content'
        ]
        
        for line in lines:
            line_clean = line.strip()
            
            # Skip empty lines (but preserve them in output)
            if not line_clean:
                pure_lines.append('')
                continue
            
            # Check for contamination
            is_contaminated = any(keyword in line_clean.lower() for keyword in contamination_keywords)
            
            # Also check for non-Mermaid patterns
            looks_like_explanation = (
                line_clean.endswith('：') or 
                line_clean.endswith(':') and not self._is_valid_mermaid_colon_usage(line_clean, diagram_type) or
                line_clean.startswith('*') or
                line_clean.startswith('-') and not self._is_valid_mermaid_dash_usage(line_clean, diagram_type)
            )
            
            if not is_contaminated and not looks_like_explanation:
                pure_lines.append(line)
        
        # Join and do final validation
        result = '\n'.join(pure_lines).strip()
        
        # Ensure it starts with a valid diagram declaration
        if not any(result.lower().startswith(dt) for dt in ['mindmap', 'flowchart', 'graph', 'sequencediagram', 'gantt', 'pie']):
            self.logger.warning("Generated syntax doesn't start with valid diagram type")
            return ""
        
        return result

    def _is_valid_mermaid_colon_usage(self, line: str, diagram_type: str) -> bool:
        """Check if colon usage is valid for Mermaid syntax"""
        if diagram_type == 'gantt':
            # Gantt charts use colons for task definitions
            return True
        elif diagram_type == 'pie':
            # Pie charts use colons in data definitions
            return '"' in line and line.count(':') == 1
        # For other types, colons at end of line usually indicate explanatory text
        return not line.strip().endswith(':')

    def _is_valid_mermaid_dash_usage(self, line: str, diagram_type: str) -> bool:
        """Check if dash usage is valid for Mermaid syntax"""
        # Dashes in flowcharts are for connections (-->)
        if '-->' in line or '->' in line:
            return True
        # Single dashes at start usually indicate list items (not Mermaid)
        return not line.strip().startswith('- ')

    def _basic_syntax_check(self, syntax: str, diagram_type: str) -> bool:
        """Basic syntax validation"""
        if not syntax or len(syntax.split('\n')) < 2:
            return False
            
        first_line = syntax.split('\n')[0].strip().lower()
        
        valid_starts = {
            'mindmap': ['mindmap'],
            'flowchart': ['flowchart', 'graph'],
            'sequencediagram': ['sequencediagram'],
            'gantt': ['gantt'],
            'pie': ['pie']
        }
        
        expected_starts = valid_starts.get(diagram_type, [])
        starts_correctly = any(first_line.startswith(start) for start in expected_starts)
        
        # FIXED: Add specific validation for mindmap
        if diagram_type == 'mindmap' and starts_correctly:
            lines = syntax.split('\n')
            # Check for root node
            has_root = any('root((' in line and '))' in line for line in lines)
            return has_root
        
        return starts_correctly

    def _create_contextual_fallback(self, diagram_type: str, query: str, kb_content: str) -> str:
        """Create contextual fallback based on actual content"""
        template = self.fallback_templates.get(diagram_type, self.fallback_templates['flowchart'])
        
        # Try to customize template with actual content
        if diagram_type == 'mindmap' and (query or kb_content):
            # Extract key terms for mindmap
            text = f"{query} {kb_content}"
            # Simple keyword extraction
            important_terms = re.findall(r'[\u4e00-\u9fff]{2,4}|[A-Za-z]{3,8}', text)[:6]
            if important_terms and len(important_terms) >= 2:
                root_topic = important_terms[0] if important_terms else "核心概念"
                branches = important_terms[1:4] if len(important_terms) > 1 else ["分支1", "分支2"]
                
                return f"""mindmap
  root(({root_topic}))
    {branches[0] if len(branches) > 0 else "分支1"}
      子项1
      子项2
    {branches[1] if len(branches) > 1 else "分支2"}
      子项3
      子项4"""
        
        return template

    async def generate_diagram_with_mcp(self, mermaid_syntax: str, filename: str = None) -> dict:
        """Generate diagram using MCP with proper error handling - FIXED for mcp-mermaid-enhanced"""
        
        # CRITICAL: Ensure absolutely pure Mermaid syntax before sending to MCP
        cleaned_syntax = self._ensure_pure_mermaid(mermaid_syntax, self.extract_diagram_type("", mermaid_syntax))
        
        if not cleaned_syntax:
            self.logger.error("No valid Mermaid syntax to send to MCP")
            return await self._fallback_diagram_generation(mermaid_syntax, filename)
        
        # Log what we're actually sending to MCP for debugging
        self.logger.info(f"Sending to MCP: {cleaned_syntax[:200]}...")
        
        if not MCP_AVAILABLE:
            return await self._fallback_diagram_generation(cleaned_syntax, filename)
        
        if not filename:
            filename = f"diagram_{uuid.uuid4().hex[:8]}"
        
        try:
            server_params = StdioServerParameters(
                command="node",
                args=["/home/enine/prime/mcp-mermaid-enhanced/build/index.js"]
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    
                    mermaid_tool = self._find_mermaid_tool(tools_result.tools)
                    if not mermaid_tool:
                        raise Exception("No suitable Mermaid tool found")
                    
                    # Prepare arguments for MCP tool - mcp-mermaid-enhanced style
                    tool_args = {
                        'mermaid': cleaned_syntax,  # Note: parameter name is 'mermaid', not 'mermaidCode'
                        'theme': 'default',
                        'backgroundColor': 'white',
                        'outputType': 'png',
                        'saveToFile': f"/tmp/{filename}.png"  # Custom path for saving
                    }
                    
                    result = await asyncio.wait_for(
                        session.call_tool(mermaid_tool.name, tool_args), 
                        timeout=45.0
                    )
                    
                    return await self._process_mcp_result_enhanced(result, filename, cleaned_syntax)
        
        except Exception as e:
            self.logger.error(f"MCP diagram generation failed: {str(e)}")
            return await self._fallback_diagram_generation(cleaned_syntax, filename)

    async def _process_mcp_result_enhanced(self, result, filename: str, mermaid_syntax: str) -> dict:
        """Process MCP result for mcp-mermaid-enhanced - FIXED VERSION"""
        
        # mcp-mermaid-enhanced automatically saves files and returns file paths, not base64
        # The response typically looks like: "PNG diagram saved to: /path/to/file.png"
        
        generated_file_path = None
        
        # Check result content for file path information
        if hasattr(result, 'content') and result.content:
            for content in result.content:
                if hasattr(content, 'text') and content.text:
                    text = content.text
                    self.logger.info(f"MCP response: {text}")
                    
                    # Extract file path from response text
                    # Pattern: "PNG diagram saved to: /path/to/file.png"
                    path_patterns = [
                        r'saved to:\s*([^\s\n]+\.(?:png|svg))',
                        r'generated:\s*([^\s\n]+\.(?:png|svg))',
                        r'created:\s*([^\s\n]+\.(?:png|svg))',
                        r'/[^\s\n]*\.(?:png|svg)'  # Any absolute path with png/svg extension
                    ]
                    
                    for pattern in path_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            generated_file_path = match.group(1) if match.groups() else match.group(0)
                            break
        
        # If no path found in response, check common locations
        if not generated_file_path:
            # mcp-mermaid-enhanced auto-generates filenames with timestamps
            # Check for recently created files
            possible_dirs = [
                "/tmp/",
                "/home/enine/prime/mcp-mermaid-enhanced/",
                os.getcwd() + "/"
            ]
            
            for directory in possible_dirs:
                try:
                    if os.path.exists(directory):
                        # Look for recently created mermaid files
                        files = [f for f in os.listdir(directory) 
                                if f.startswith('mermaid-') and f.endswith('.png')]
                        if files:
                            # Get most recently created
                            files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)), reverse=True)
                            generated_file_path = os.path.join(directory, files[0])
                            break
                except Exception:
                    continue
        
        # Also check our custom path
        custom_path = f"/tmp/{filename}.png"
        if not generated_file_path and os.path.exists(custom_path):
            generated_file_path = custom_path
        
        # If we found a file, read and encode it
        if generated_file_path and os.path.exists(generated_file_path):
            try:
                with open(generated_file_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                self.logger.info(f"Successfully read generated file: {generated_file_path}")
                
                # Clean up the file (optional - you might want to keep it)
                try:
                    os.unlink(generated_file_path)
                    self.logger.info(f"Cleaned up temporary file: {generated_file_path}")
                except Exception as e:
                    self.logger.warning(f"Could not clean up file {generated_file_path}: {e}")
                
                return {
                    'success': True,
                    'image_base64': img_base64,
                    'filename': f"{filename}.png",
                    'mermaid_syntax': mermaid_syntax,
                    'method': 'mcp_enhanced_file',
                    'original_path': generated_file_path
                }
                
            except Exception as e:
                self.logger.error(f"Error reading generated file {generated_file_path}: {e}")
        
        # If we get here, MCP succeeded but we couldn't find/read the file
        self.logger.warning("MCP succeeded but couldn't locate or read generated file")
        
        return {
            'success': True,
            'image_base64': None,
            'error': 'MCP processed successfully but could not locate generated image file',
            'mermaid_syntax': mermaid_syntax,
            'filename': f"{filename}.png",
            'method': 'mcp_enhanced_syntax_only'
        }

    def _find_mermaid_tool(self, tools):
        """Find appropriate Mermaid tool from available tools - Updated for mcp-mermaid-enhanced"""
        # mcp-mermaid-enhanced uses 'generate_mermaid_diagram' as the tool name
        preferred_names = ['generate_mermaid_diagram', 'generate_mermaid', 'mermaid', 'create_diagram', 'render_mermaid']
        
        for preferred in preferred_names:
            for tool in tools:
                if preferred.lower() in tool.name.lower():
                    self.logger.info(f"Found MCP tool: {tool.name}")
                    return tool
        
        # Fallback to first available tool
        if tools:
            self.logger.info(f"Using fallback tool: {tools[0].name}")
            return tools[0]
        
        return None

    async def _fallback_diagram_generation(self, mermaid_syntax: str, filename: str = None) -> dict:
        """Fallback diagram generation using subprocess"""
        if not filename:
            filename = f"diagram_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create temporary mermaid file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
                f.write(mermaid_syntax)
                mermaid_file = f.name
            
            output_path = f"/tmp/{filename}.png"
            
            try:
                import subprocess
                # Try using mermaid CLI if available
                result = subprocess.run([
                    'mmdc', '-i', mermaid_file, '-o', output_path, 
                    '-t', 'neutral', '-b', 'white'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    with open(output_path, 'rb') as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                    
                    # Clean up
                    os.unlink(mermaid_file)
                    os.unlink(output_path)
                    
                    return {
                        'success': True,
                        'image_base64': img_base64,
                        'filename': f"{filename}.png",
                        'mermaid_syntax': mermaid_syntax,
                        'method': 'cli_fallback'
                    }
                    
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                self.logger.warning("Mermaid CLI not available or failed")
            
            # Clean up temp file
            try:
                os.unlink(mermaid_file)
            except:
                pass
            
        except Exception as e:
            self.logger.error(f"Fallback generation failed: {str(e)}")
        
        # Return syntax-only success
        return {
            'success': True,
            'image_base64': None,
            'error': 'No image generation method available',
            'mermaid_syntax': mermaid_syntax,
            'filename': f"{filename}.png",
            'method': 'syntax_only'
        }
    
def get_next_pages_direct_query(rag_client, primary_doc, collection_name, max_additional_pages=1):
    """Get the very next page numbers from the same document - optimized"""
    primary_doc_name = getattr(primary_doc, 'document_name', '')
    primary_page = getattr(primary_doc, 'page_number', None)
    
    if not primary_doc_name or primary_page is None:
        return [primary_doc]
    
    pages_found = [primary_doc]
    
    # Optimized search queries - fewer, more targeted
    base_queries = [
        f"{primary_doc_name}",
        "continue"
    ]
    
    for i in range(1, max_additional_pages + 1):
        next_page_number = primary_page + i
        next_page_doc = None
        
        for query in base_queries:
            try:
                results = rag_client.search(
                    query=query,
                    collection_name=collection_name,
                    top_k=3# Reduced from 10
                )
                
                if hasattr(results, 'results') and results.results:
                    for doc in results.results:
                        doc_name = getattr(doc, 'document_name', '')
                        doc_page = getattr(doc, 'page_number', None)
                        
                        if doc_name == primary_doc_name and doc_page == next_page_number:
                            next_page_doc = doc
                            break
                
                if next_page_doc:
                    break
                    
            except Exception:
                continue
        
        if next_page_doc:
            pages_found.append(next_page_doc)
        else:
            break  # Stop if we can't find consecutive pages
    
    return pages_found


def get_multi_document_pages(rag_client, filtered_results: List, collection_name: str, max_docs: int = 3, pages_per_doc: int = 2) -> List[Dict]:
    """
    Get multiple pages from top documents for comprehensive coverage
    """
    all_pages = []
    doc_groups = {}
    
    # Group results by document
    for result in filtered_results[:max_docs * pages_per_doc]:  # Get more than needed initially
        doc_name = getattr(result, 'document_name', '')
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append(result)
    
    # Take top documents and get their pages
    processed_docs = 0
    for doc_name, doc_results in doc_groups.items():
        if processed_docs >= max_docs:
            break
            
        # Sort by score and take top pages for this document
        doc_results.sort(key=lambda x: x.normalized_score, reverse=True)
        pages_to_take = min(pages_per_doc, len(doc_results))
        
        for i in range(pages_to_take):
            result = doc_results[i]
            page_num = getattr(result, 'page_number', None)
            
            # Try to get adjacent pages for better context
            adjacent_pages = []
            if page_num and i == 0:  # Only for the top page of each document
                try:
                    # Get the next page if it exists
                    next_page_results = rag_client.search(
                        query=f"document:{doc_name} page:{page_num + 1}",
                        collection_name=collection_name,
                        top_k=1
                    )
                    if next_page_results.results:
                        adjacent_pages.append(next_page_results.results[0])
                except:
                    pass  # Continue without adjacent page if search fails
            
            all_pages.append(result)
            all_pages.extend(adjacent_pages)
        
        processed_docs += 1
    
    return all_pages

def combine_images_for_vlm(page_docs):
    """Prepare multiple page images for VLM processing - highly optimized"""
    combined_images = []
    
    for doc in page_docs:
        if not hasattr(doc, 'img_base64') or not doc.img_base64:
            continue
        
        # Skip expensive validation for speed - just basic checks
        base64_data = doc.img_base64
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]
        
        # Quick format detection from first few bytes
        try:
            decoded_start = base64.b64decode(base64_data[:20])
            if decoded_start.startswith(b'\xff\xd8'):
                image_format = 'JPEG'
            elif decoded_start.startswith(b'\x89PNG'):
                image_format = 'PNG'
            else:
                image_format = 'unknown'
        except:
            image_format = 'unknown'
        
        combined_images.append({
            'base64': base64_data,
            'page_number': getattr(doc, 'page_number', 'unknown'),
            'document_name': getattr(doc, 'document_name', 'unknown'),
            'format': image_format,
            'score': getattr(doc, 'normalized_score', 0)
        })
    
    return combined_images


def enhance_multi_document_prompt(query: str, images_data: List[Dict]) -> str:
    """
    Generate a streamlined prompt for accurate document extraction.
    Focus: Clear instructions, flexible handling, maintainable structure.
    """
    
    # Initialize processor
    try:
        processor = query_processor
    except NameError:
        processor = ModernQueryProcessor()
    
    features = processor.extract_semantic_features(query)
    
    # Simplified document grouping
    doc_groups = {}
    valid_docs = 0
    
    for img in images_data:
        name = img.get('document_name', f'Document_{valid_docs + 1}')
        page = img.get('page_number')
        
        if page is not None:
            try:
                page_num = int(page)
                doc_groups.setdefault(name, []).append(page_num)
                valid_docs += 1
            except (ValueError, TypeError):
                continue
    
    # Create simple source reference
    sources_info = []
    for name, pages in doc_groups.items():
        sorted_pages = sorted(set(pages))
        page_range = f"p.{sorted_pages[0]}" if len(sorted_pages) == 1 else f"pp.{min(sorted_pages)}-{max(sorted_pages)}"
        sources_info.append(f"• {name} ({page_range})")
    
    sources_text = "Available sources:\n" + "\n".join(sources_info)
    
    # Enhanced query type detection with comprehensive Chinese support
    def get_query_intent(query_text: str) -> str:
        query_lower = query_text.lower()
        
        # Table generation indicators (Chinese + English)
        table_keywords = [
            # Chinese variants
            '生成表格', '制作表格', '创建表格', '建立表格', '构建表格', 
            '整理表格', '汇总表格', '制表', '表格化', '表格形式', 
            '表格格式', '表格显示', 'markdown表格',
            # English variants
            'create table', 'generate table', 'make table', 'build table', 
            'table format', 'tabulate'
        ]
        if any(keyword in query_lower for keyword in table_keywords):
            return 'table_creation'
        
        # Data extraction indicators (Chinese + English)
        data_keywords = [
            # Quantity Chinese
            '多少', '几个', '几种', '几项', '数量', '总数', '合计', '总计',
            # Financial Chinese  
            '金额', '价格', '成本', '费用', '收入', '支出', '利润', '损失',
            # Statistical Chinese
            '数据', '数值', '平均', '最大', '最小', '最高', '最低', '标准', '限额',
            # Question words
            '什么是', '多大', '多长', '多重', '有多少',
            # English variants
            'how many', 'what is the', 'how much', 'what are the', 
            'total number', 'amount of', 'price of', 'cost of'
        ]
        if any(keyword in query_lower for keyword in data_keywords):
            return 'data_extraction'
        
        # Section-specific indicators (Chinese + English)
        section_keywords = [
            # Chinese section markers
            '第', '条', '款', '章', '节', '项', '号', '段',
            '第一', '第二', '第三', '第四', '第五', '第六', '第七', '第八', '第九', '第十',
            '条款', '章节', '段落', '部分', '小节',
            # English variants
            'section', 'article', 'clause', 'paragraph', 'chapter', 
            'part', 'subsection'
        ]
        if any(keyword in query_lower for keyword in section_keywords):
            return 'section_specific'
        
        # List content indicators (Chinese + English)
        list_keywords = [
            # Chinese list words
            '列出', '列举', '罗列', '枚举', '列表', '清单', 
            '包括', '包含', '含有', '有哪些', '哪些', 
            '步骤', '流程', '程序', '过程', '方法',
            # English variants
            'list', 'include', 'contain', 'enumerate', 'what are', 
            'steps', 'process', 'procedure', 'methods'
        ]
        if any(keyword in query_lower for keyword in list_keywords):
            return 'list_content'
        
        return 'general_info'
    
    query_intent = get_query_intent(query)
    
    # Generate appropriate instructions based on intent (bilingual)
    def get_extraction_instructions(intent: str) -> str:
        instructions = {
            'table_creation': """
## 表格提取指令 | Table Extraction Instructions

### 中文指令：
1. **定位源表格** - 在文档中找到要提取的表格
2. **逐个单元格提取** - 读取每个单元格的实际内容
3. **保持结构** - 维持原始行列结构
4. **处理空单元格** - 空白格子在输出中保持空白
5. **使用标准markdown格式** 

**输出格式：**
```
| 表头1 | 表头2 | 表头3 |
|-------|-------|-------|
| 数据1 | 数据2 |       |
|       | 数据4 | 数据5 |
```

### English Instructions:
1. **Locate the source table** in the documents
2. **Extract cell by cell** - read each cell's actual content  
3. **Preserve structure** - maintain original rows and columns
4. **Handle empty cells** - leave blank cells empty in output
5. **Use standard markdown** table format

**核心要求 | Key Requirements:**
- 完全匹配原表格结构 | Match original table structure exactly
- 不解释或填充空单元格 | No interpretation or filling of empty cells  
- 按原样保留所有可见内容 | Preserve all visible content as-is
""",
            
            'data_extraction': """
## 数据提取指令 | Data Extraction Instructions

### 中文指令：
1. **查找相关数据** - 在源文档中定位相关信息
2. **提取确切数值** - 不近似或计算，提取原始数据
3. **包含上下文** - 说明数据来自哪个文档/页面
4. **验证准确性** - 核对数字和数据

### English Instructions:
1. **Find relevant data** in the source documents
2. **Extract exact values** - no approximation or calculation
3. **Include context** - mention which document/page contains the data
4. **Verify accuracy** - cross-check numbers and figures

**输出格式 | Output Format:**
- 直接回答问题 | Direct answer to the question
- 提供源文档支持数据 | Supporting data from sources
- 清楚标明来源位置 | Clear indication of source location
""",
            
            'section_specific': """
## 章节提取指令 | Section-Specific Extraction Instructions

### 中文指令：
1. **定位精确章节** - 找到查询中提及的确切章节
2. **提取完整内容** - 提取该章节的所有内容
3. **保持原格式** - 维持原始格式和结构
4. **包含子章节** - 如果存在子章节也要包含

### English Instructions:  
1. **Locate exact section** mentioned in query
2. **Extract complete content** of that section
3. **Maintain original formatting** and structure
4. **Include subsections** if they exist

**如果找不到章节 | If Section Not Found:**
- 明确说明"未找到章节" | Clearly state "Section not found"
- 建议最接近的可用章节 | Suggest closest available sections
- 不要近似或猜测 | Do not approximate or guess
""",
            
            'list_content': """
## 列表内容提取指令 | List Content Extraction Instructions

### 中文指令：
1. **找到相关列表** - 在文档中定位相关列表
2. **保持原格式** - 保留原始格式（项目符号、数字等）
3. **包含所有项目** - 不要总结或截断
4. **维持层次结构** - 保持缩进和子项目

### English Instructions:
1. **Find the relevant list** in the documents
2. **Preserve original format** (bullets, numbers, etc.)
3. **Include all items** - don't summarize or truncate
4. **Maintain hierarchy** - keep indentation and sub-items

**输出格式 | Output Format:**
- 完全按原样复制列表 | Reproduce the list exactly as shown
- 保持原编号/项目符号样式 | Keep original numbering/bullet style
- 维持项目顺序和分组 | Maintain item order and grouping
""",
            
            'general_info': """
## 一般信息提取指令 | General Information Extraction Instructions

### 中文指令：
1. **识别相关内容** - 在所有文档中识别相关内容
2. **综合信息** - 适当时综合多个来源的信息
3. **保持准确性** - 不推测或假设
4. **提供上下文** - 说明哪些来源包含哪些信息

### English Instructions:
1. **Identify relevant content** across all documents
2. **Synthesize information** from multiple sources when appropriate
3. **Maintain accuracy** - don't interpolate or assume
4. **Provide context** - indicate which sources contain which information

**输出格式 | Output Format:**
- 全面回答查询问题 | Comprehensive answer addressing the query
- 清楚归属源文档 | Clear attribution to source documents
- 有条理地呈现信息 | Organized presentation of information
"""
        }
        
        return instructions.get(intent, instructions['general_info'])
    
    extraction_instructions = get_extraction_instructions(query_intent)
    
    # Language detection helper
    def detect_language(text: str) -> str:
        """Detect if query is primarily Chinese or English"""
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
        return 'chinese' if chinese_chars > english_chars else 'english'
    
    query_language = detect_language(query)
    
    # Consistent output format based on query intent and language
    def get_output_format(intent: str, language: str) -> str:
        if intent == 'table_creation':
            if language == 'chinese':
                return """
## 输出要求

### 表格生成格式：
**直接回答：** 立即以markdown格式呈现提取的表格，然后提供最少的上下文。

**结构：**
1. **生成的表格** - 完整的markdown格式表格
2. **来源信息** - 使用自然段落格式简述来源位置

### 格式指导：
- **主要内容：** 以markdown表格为主要响应内容
- **简洁说明：** 仅提供必要的源文档说明
- **避免重复：** 不要在支持信息中重新生成表格
- **专业格式：** 使用标题-段落格式，避免"项目 内容"格式
- **保持原样：** 不对已有结构信息添加项目符号或编号

### 质量标准：
- **准确性：** 完全按源文档提取表格内容
- **简洁性：** 避免冗长结论和总结
- **一致性：** 保持响应结构简洁统一
- **专业性：** 使用清晰的标题-段落格式表述
"""
            else:
                return """
## Output Requirements

### TABLE GENERATION FORMAT:
**Direct Answer:** Present the extracted table immediately in markdown format, followed by minimal context.

**Structure:**
1. **Generated Table** - The complete table in markdown format
2. **Source Information** - Brief statement of source location using natural paragraph format

### Formatting Guidelines:
- **Main Content:** Present markdown table as primary response content
- **Brief Explanation:** Provide only necessary source document information
- **Avoid Repetition:** Don't regenerate table in supporting information
- **Professional Format:** Use title-paragraph format, avoid "item content" format
- **Preserve Structure:** Don't add bullets or numbering to already structured information

### Quality Standards:
- **Accuracy:** Extract table content exactly as shown in source documents
- **Conciseness:** Avoid lengthy conclusions and summaries
- **Consistency:** Maintain unified response structure
- **Professionalism:** Use clear title-paragraph formatting
"""
        else:
            if language == 'chinese':
                return """
## 输出要求

### 回答结构：
**直接回答**
[以自然段落格式提供用户问题的具体答案]

**支持信息**  
[使用标题-段落格式包含源文档的相关详细信息并明确归属]

### 格式指导：
- **自然语言：** 使用清晰的标题-段落格式
- **避免表格式：** 不使用"项目 内容"的表格式表述
- **保持专业：** 使用标准的学术/商务写作格式
- **源文档归属：** 清楚标明信息来源
- **保持原样：** 不对已有结构信息添加项目符号或编号

### 质量标准：
- **准确性：** 完全按源文档所示提取内容
- **完整性：** 解决用户问题的所有方面  
- **清晰性：** 使用专业的标题-段落结构
- **一致性：** 保持统一的响应格式
"""
            else:
                return """
## Output Requirements

### Answer Structure:
**Direct Answer**
[Provide the specific answer to the user's question in natural paragraph format]

**Supporting Information**  
[Include relevant details from source documents with clear attribution using title-paragraph format]

### Formatting Guidelines:
- **Natural Language:** Use clear title-paragraph format
- **Avoid Tabular Format:** Don't use "item content" tabular presentation
- **Professional Standard:** Use academic/business writing format
- **Source Attribution:** Clearly indicate information sources
- **Preserve Structure:** Don't add bullets or numbering to already structured information

### Quality Standards:
- **Accuracy:** Extract content exactly as shown in source documents
- **Completeness:** Address all aspects of the user's question  
- **Clarity:** Use professional title-paragraph structure
- **Consistency:** Maintain uniform response format
"""
    
    output_format = get_output_format(query_intent, query_language)
    
    # Build final prompt with strict formatting enforcement
    def get_consistency_guidelines(language: str) -> str:
        if language == 'chinese':
            return """
## 严格格式要求：

### 绝对禁止的格式：
- **严格禁止：** 任何情况下都不得使用"项目-内容"表格结构
- **严格禁止：** 不得使用任何两列表格来呈现信息
- **严格禁止：** 不得使用"| 项目 | 内容 |"或类似的表格格式
- **严格禁止：** 不得创建任何形式的结构化表格来展示支持信息
- **严格禁止：** 不得对已有编号或项目符号的内容添加额外的编号或符号
- **严格禁止：** 不得改变原文档中现有的编号系统或层次结构

### 必须遵循的原则：
- **保持原始结构：** 完全按照源文档的编号、项目符号和格式呈现
- **不添加格式：** 如果原文有编号1.2.3.，保持原样，不要添加新的编号或符号
- **不改变层次：** 保持原文档的缩进和层次关系
- **保持符号：** 原文的冒号(:)、项目符号(•)等格式符号必须原样保留

### 正确的处理方式：
**对于已有结构的内容（如带冒号的标题和编号列表）：**
- 完全按原样复制，不添加任何新的格式
- 保持原有的编号顺序和符号系统
- 维持原始的缩进和层次关系

### 错误的处理方式（绝对不能做）：
- 对已有编号的内容再次添加编号
- 改变原有的项目符号样式
- 添加不存在的结构化格式
- 重新组织已有的层次结构

### 响应质量要求：
- **忠实复制：** 完全按源文档的格式呈现内容
- **格式保真：** 不对原有格式进行任何修改
- **结构保持：** 维持原始文档的组织方式
- **准确提取：** 逐字逐句按原样提取内容
"""
        else:
            return """
## STRICT FORMATTING REQUIREMENTS:

### ABSOLUTELY PROHIBITED FORMATS:
- **STRICTLY FORBIDDEN:** Never use "Item-Content" table structure under any circumstances
- **STRICTLY FORBIDDEN:** Never use any two-column tables to present information
- **STRICTLY FORBIDDEN:** Never use "| Item | Content |" or similar table formats
- **STRICTLY FORBIDDEN:** Never create any form of structured tables for supporting information
- **STRICTLY FORBIDDEN:** Never add numbering or bullets to content that already has numbering or bullets
- **STRICTLY FORBIDDEN:** Never alter existing numbering systems or hierarchical structures

### MANDATORY PRINCIPLES:
- **Preserve Original Structure:** Present content exactly as formatted in source documents
- **No Added Formatting:** If source has numbering 1.2.3., keep as-is, don't add new numbers or symbols
- **Maintain Hierarchy:** Keep original document indentation and hierarchical relationships
- **Keep Symbols:** Preserve original colons (:), bullet points (•), and other formatting symbols exactly

### CORRECT HANDLING METHOD:
**For Already Structured Content (like titles with colons and numbered lists):**
- Copy exactly as shown, without adding any new formatting
- Preserve original numbering sequence and symbol system
- Maintain original indentation and hierarchical relationships

### INCORRECT HANDLING (ABSOLUTELY FORBIDDEN):
- Adding numbering to already numbered content
- Changing existing bullet point styles
- Adding structural formatting that doesn't exist
- Reorganizing existing hierarchical structures

### Response Quality Requirements:
- **Faithful Reproduction:** Present content exactly as formatted in source documents
- **Format Preservation:** Make no modifications to existing formatting
- **Structure Maintenance:** Keep original document organization intact
- **Accurate Extraction:** Extract content word-for-word as originally shown
"""
    
    consistency_guidelines = get_consistency_guidelines(query_language)
    
    # Build final prompt
    return f"""# Document Analysis Task

## Query: {query}

{sources_text}

{extraction_instructions}

{output_format}

{consistency_guidelines}

---
**CRITICAL FORMATTING RULES:**
1. **NEVER add numbering or bullets to already numbered/bulleted content**
2. **PRESERVE original formatting exactly** - if source has "1. 落实相关制度与管理规定:", keep it exactly as "1. 落实相关制度与管理规定:"
3. **MAINTAIN original hierarchy** - don't change indentation or structure
4. **NO "项目-内容" tables** under any circumstances
5. **COPY formatting symbols** like colons (:) and bullets (•) exactly as shown

**ABSOLUTE PRIORITY: Extract and present content with 100% formatting fidelity to the source documents.**
"""

def call_ollama_vlm_multipage(ollama_url: str, model: str, prompt: str, images_data: List[Dict], timeout: int = 300) -> Tuple[str, Dict]:
    """Call Ollama VLM with optimized settings for speed"""
    image_base64_list = []
    for img in images_data:
        base64_data = img['base64']
        
        # Remove data URL prefix if present
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]
        
        # Quick validation - just check if it looks like base64
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', base64_data.replace('\n', '')):
            logger.error(f"Invalid base64 format detected")
            raise ValueError("Invalid base64 image data format")
        
        image_base64_list.append(base64_data)
    
    # Clean prompt efficiently
    cleaned_prompt = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', prompt)
    
    # Optimized settings for speed vs quality balance
    ollama_payload = {
        "model": model,
        "prompt": cleaned_prompt,
        "images": image_base64_list,
        "stream": False,
        "options": {
            "temperature": 0.4,          # Slightly higher to promote diversity
            "top_p": 0.9,                # Allows more token variety
            "repeat_penalty": 2.0,       # Strong penalty to avoid loops
            "presence_penalty": 0.6,     # Encourage new concepts
            "frequency_penalty": 0.6,    # Discourage exact repeats
            "num_ctx": 6144,
            "num_predict": 3072,
            "num_thread": -1,
            "num_gpu": -1,
            "seed": 42
         }

    }
    
    ollama_full_url = f"{ollama_url.rstrip('/')}/api/generate"
    
    # Reduced retries for speed
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Ollama request attempt {attempt + 1}")
            
            response = requests.post(
                ollama_full_url,
                json=ollama_payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    raw_response = response_json.get('response', '')
                    
                    # Minimal cleaning for speed
                    clean_response = re.sub(r'^(?:根据文档内容|根据查询|在文档中|文档显示)[，:：]\s*', '', raw_response)
                    clean_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_response)
                    
                    return clean_response, response_json
                except ValueError as json_error:
                    logger.error(f"JSON parse error: {json_error}")
                    if attempt == max_retries:
                        raise Exception(f"Invalid JSON response: {json_error}")
            else:
                logger.error(f"Ollama error {response.status_code}")
                if attempt == max_retries:
                    raise Exception(f"Ollama error {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout on attempt {attempt + 1}")
            if attempt == max_retries:
                raise Exception("Request timed out")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {str(e)}")
            if attempt == max_retries:
                raise Exception(f"Network failure: {str(e)}")
        
        # Shorter backoff time
        if attempt < max_retries:
            wait_time = 1 + attempt  # Linear backoff instead of exponential
            time.sleep(wait_time)
    
    raise Exception("Unexpected error in Ollama request")

def post_process_response(response_text: str, original_query: str, doc_names: List[str]) -> str:
    """
    Simplified post-processing with enhanced table method - OPTIMIZED
    """
    # Remove AI prefixes using compiled patterns
    for pattern in _PATTERNS.ai_prefixes:
        response_text = pattern.sub('', response_text, count=1)
    
    # Process tables if they exist and aren't already properly formatted
    if not table_processor.is_already_markdown_table(response_text):
        response_text = table_processor.process_tables_in_response(response_text)
    
    # Clean up formatting using compiled pattern
    response_text = _PATTERNS.newline_cleanup.sub('\n\n', response_text)
    response_text = response_text.strip()
    
    # Add source reference
    if doc_names and not _PATTERNS.source_check.search(response_text):
        doc_ref = "、".join(set(doc_names))
        response_text += f"\n\n**来源**：《{doc_ref}》"
    
    return response_text

        
def ensure_collection_exists(rag_client, collection_name: str):
    """Ensure ColiVara collection exists"""
    try:
        # Try to create collection (ColiVara will handle if it already exists)
        rag_client.create_collection(name=collection_name)
        logger.info(f"Collection ensured: {collection_name}")
    except Exception as e:
        # Collection might already exist, which is fine
        logger.debug(f"Collection creation note: {e}")

def ensure_bucket_exists(bucket_name: str):
    """Ensure MinIO bucket exists"""
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logger.info(f"Created bucket: {bucket_name}")
    except S3Error as e:
        logger.error(f"Error creating bucket {bucket_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create bucket: {e}")

async def general_knowledge_fallback(query: str) -> QueryResponse:
    """Fallback to general knowledge when KB fails"""
    try:
        prompt = f"你是一个专业助手。请根据你的知识回答：{query}"
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '')
            return QueryResponse(
                result=result,
                table_data=None,
                source_info={"source": "general_knowledge"},
                debug_info=["Used general knowledge fallback"],
                images_info=[],
                base64_preview=None,
                full_response=response.json()
            )
    except Exception as e:
        logger.error(f"General knowledge fallback failed: {str(e)}")
    
    return QueryResponse(
        result="无法找到相关文档信息，且通用知识库查询失败",
        source_info={"source": "error"},
        debug_info=["General knowledge fallback failed"],
        images_info=[]
    )

def get_system_info():
    """Get system information for metadata"""
    return {
        "platform": sys.platform,
        "python_version": sys.version,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }


# Initialize the Mermaid agent
mermaid_agent = MermaidAgent()

@app.post("/query", response_model=QueryResponse,
          summary="Query documents",
          description="**Example Request:**\n```json\n{\n  \"api_key\": \"your_api_key\",\n  \"query\": \"开个个人独资企业有什么要求？\",\n  \"include_next_page\": false,\n  \"max_additional_pages\": 1,\n  \"allow_general_fallback\": true,\n  \"similarity_threshold\": 0.25\n}\n```")
async def query_documents(request: QueryRequest):
    """Enhanced query handling with improved Mermaid diagram generation using actual document images"""
    debug_info = []
    
    try:
        debug_info.append(f"Starting query: {request.query}")
        debug_info.append(f"Similarity threshold: {request.similarity_threshold}")
        debug_info.append(f"Collection: {DEFAULT_COLLECTION}")

        # Step 1: Perform expanded fuzzy search
        expanded_query = expand_query_terms(request.query)
        debug_info.append(f"Expanded query: {expanded_query}")
        
        # Step 2: Perform ColiVara search
        rag_client = ColiVara(api_key=request.api_key, base_url=COLIVARA_BASE_URL_QUERY)
        
        search_top_k = 15
        
        results = await asyncio.to_thread(
            rag_client.search,
            query=expanded_query,
            collection_name=DEFAULT_COLLECTION,
            top_k=search_top_k
        )
        
        # Enhanced fallback logic
        if not results.results:
            debug_info.append("No documents found in search results")
            if request.allow_general_fallback:
                debug_info.append("Using general knowledge fallback")
                return await general_knowledge_fallback(request.query)
            else:
                raise HTTPException(
                    status_code=404,
                    detail="No matching documents found and general fallback disabled"
                )
        
        # Filter results by similarity threshold
        filtered_results = [doc for doc in results.results 
                          if doc.normalized_score >= request.similarity_threshold]
        
        if not filtered_results:
            debug_info.append(f"No documents meet similarity threshold ({request.similarity_threshold})")
            if request.allow_general_fallback:
                debug_info.append("Using general knowledge fallback")
                return await general_knowledge_fallback(request.query)
            else:
                raise HTTPException(
                    status_code=404,
                    detail="No documents meet similarity threshold and general fallback disabled"
                )
        
        # Boost relevant sections - using proper ColiVara attributes
        for doc in filtered_results:
            # ColiVara docs might have different text attributes - check what's available
            doc_text = ""
            for attr in ['content', 'text_content', 'extracted_text', 'text']:
                if hasattr(doc, attr):
                    doc_text = getattr(doc, attr, '')
                    break
            
            if is_relevant_section(doc_text, request.query):
                doc.normalized_score = min(doc.normalized_score * 1.5, 1.0)
                debug_info.append(f"Boosted document: {getattr(doc, 'document_name', '')} (page {getattr(doc, 'page_number', '?')})")
        
        # Re-sort by boosted score
        filtered_results = sorted(filtered_results, key=lambda x: x.normalized_score, reverse=True)
        
        # Get multi-document pages
        max_docs = 3
        pages_per_doc = 2
        
        if request.include_next_page:
            page_docs = get_multi_document_pages(
                rag_client, filtered_results, DEFAULT_COLLECTION, max_docs, pages_per_doc
            )
            debug_info.append(f"Retrieved {len(page_docs)} pages from up to {max_docs} documents")
        else:
            page_docs = []
            seen_docs = set()
            for doc in filtered_results:
                doc_name = getattr(doc, 'document_name', '')
                if doc_name not in seen_docs and len(page_docs) < max_docs:
                    page_docs.append(doc)
                    seen_docs.add(doc_name)
            debug_info.append(f"Using single-page mode with {len(page_docs)} documents")
        
        # Get images for VLM processing
        images_data = combine_images_for_vlm(page_docs)
        
        # Initialize diagram variables
        diagram_base64_for_frontend = None
        
        # FIXED DIAGRAM GENERATION LOGIC - Now properly uses document images
        try:
            if not images_data:
                debug_info.append("No valid image data found - cannot generate diagrams")
                should_generate_diagram = False
            else:
                # Use the MermaidAgent's improved detection with actual document context
                debug_info.append(f"Using {len(images_data)} document images for diagram detection")
                
                # Extract some text for basic detection, but we'll use images for generation
                kb_content = ""  # We'll rely on images instead of pre-extracted text
                
                should_generate_diagram = mermaid_agent.should_generate_diagram_with_context(request.query, request.query)
                debug_info.append(f"Intelligent diagram detection result: {should_generate_diagram}")
        except Exception as e:
            debug_info.append(f"MermaidAgent detection error: {e}")
            should_generate_diagram = False

        if should_generate_diagram and images_data:
            debug_info.append("Generating Mermaid diagram using document images")
            
            try:
                # Generate Mermaid syntax using the improved agent with actual document images
                mermaid_syntax = await asyncio.to_thread(
                    mermaid_agent.generate_standalone_mermaid_syntax,
                    request.query,
                    "",  # Empty kb_content since we're using images
                    images_data  # Pass the actual document images
                )
                debug_info.append(f"Generated Mermaid syntax ({len(mermaid_syntax) if mermaid_syntax else 0} chars)")
                
                if mermaid_syntax:
                    # Generate diagram using MCP with enhanced processing
                    diagram_result = await mermaid_agent.generate_diagram_with_mcp(mermaid_syntax)
                    debug_info.append(f"MCP generation result: {diagram_result.get('method', 'unknown')}")
                    
                    if diagram_result.get('success'):
                        # Process successful diagram generation
                        image_b64 = diagram_result.get('image_base64')
                        saved_path = ""
                        
                        if image_b64:
                            # We have actual image data - prepare for frontend first
                            diagram_base64_for_frontend = image_b64
                            debug_info.append("Prepared base64 data for frontend")
                            
                            # Try to save the file (but don't let saving failure affect frontend display)
                            try:
                                save_dir = "/home/enine/prime/backend/mermaid_diagrams"
                                os.makedirs(save_dir, exist_ok=True)
                                unique_name = f"mermaid_{uuid.uuid4().hex[:12]}.png"
                                saved_path = os.path.join(save_dir, unique_name)
                                
                                # Decode and save image
                                image_data = base64.b64decode(image_b64)
                                with open(saved_path, "wb") as f:
                                    f.write(image_data)
                                
                                debug_info.append(f"Diagram image saved successfully: {saved_path}")
                                
                            except Exception as save_error:
                                debug_info.append(f"Failed to save diagram image (frontend will still work): {save_error}")
                                saved_path = ""
                        else:
                            debug_info.append("Diagram generated in syntax-only mode (no image data)")
                        
                        # Create enhanced response with diagram
                        doc_groups = {}
                        for doc in page_docs:
                            doc_name = getattr(doc, 'document_name', '')
                            if doc_name not in doc_groups:
                                doc_groups[doc_name] = []
                            doc_groups[doc_name].append(doc)
                        
                        # Build response text
                        combined_text = "根据文档内容分析，生成了相关图表：\n\n"
                        
                        for doc_name, docs in doc_groups.items():
                            combined_text += f"**《{doc_name}》** 相关信息：\n"
                            for doc in docs:
                                page_num = getattr(doc, 'page_number', '')
                                combined_text += f"  第{page_num}页的内容已用于生成图表\n"
                            combined_text += "\n"
                        
                        if diagram_base64_for_frontend:
                            combined_text += "✅ 已成功生成可视化图表\n"
                        else:
                            combined_text += "📝 已生成图表语法（图像渲染暂不可用）\n"
                        
                        combined_text += f"\n**图表类型**：{mermaid_agent.extract_diagram_type(request.query, request.query)}\n"
                        
                        source_names = list(doc_groups.keys())
                        combined_text += f"\n**来源**：《{'》、《'.join(source_names)}》"
                        
                        source_info = {
                            'documents_used': source_names,
                            'pages_used': len(page_docs),
                            'primary_page': getattr(page_docs[0], 'page_number', '') if page_docs else '',
                            'diagram_generated': True,
                            'diagram_has_image': bool(diagram_base64_for_frontend),
                            'diagram_filename': diagram_result.get('filename'),
                            'mermaid_syntax': diagram_result.get('mermaid_syntax') or mermaid_syntax,
                            'diagram_saved_path': saved_path,
                            'diagram_method': diagram_result.get('method', 'unknown'),
                            'diagram_type': mermaid_agent.extract_diagram_type(request.query, request.query),
                            'images_used_count': len(images_data)
                        }
                        
                        return QueryResponse(
                            result=combined_text,
                            table_data=None,
                            source_info=source_info,
                            debug_info=debug_info,
                            images_info=[],
                            base64_preview="",
                            diagram_base64=diagram_base64_for_frontend,
                            full_response={
                                "diagram_generated": True, 
                                "diagram_has_image": bool(diagram_base64_for_frontend),
                                "mermaid_syntax": diagram_result.get('mermaid_syntax') or mermaid_syntax,
                                "diagram_type": mermaid_agent.extract_diagram_type(request.query, request.query),
                                "images_processed": len(images_data)
                            }
                        )
                    else:
                        debug_info.append(f"Diagram generation failed: {diagram_result.get('error', 'Unknown error')}")
                else:
                    debug_info.append("No valid Mermaid syntax generated from document images")
                    
            except Exception as e:
                debug_info.append(f"Error in diagram generation pipeline: {e}")
            
            # Continue with normal processing after diagram generation attempt
            debug_info.append("Continuing with normal document processing")
        
        # Continue with existing logic for non-diagram cases or when diagram generation fails
        if not images_data:
            debug_info.append("No valid image data found - using text-only fallback")
            doc_groups = {}
            for doc in page_docs:
                doc_name = getattr(doc, 'document_name', '')
                if doc_name not in doc_groups:
                    doc_groups[doc_name] = []
                doc_groups[doc_name].append(doc)
            
            combined_text = ""
            for doc_name, docs in doc_groups.items():
                combined_text += f"\n\n根据《{doc_name}》：\n"
                for doc in docs:
                    combined_text += f"第{getattr(doc, 'page_number', '')}页的相关内容\n"
            
            source_names = list(doc_groups.keys())
            combined_text += f"\n\n来源：《{'》、《'.join(source_names)}》"
            
            return QueryResponse(
                result=combined_text[:6000],
                source_info={
                    'documents_used': source_names,
                    'pages_used': len(page_docs),
                    'primary_page': getattr(page_docs[0], 'page_number', '') if page_docs else '',
                },
                debug_info=debug_info,
                images_info=[],
                diagram_base64=diagram_base64_for_frontend
            )
        
        # Prepare enhanced multi-document prompt
        enhanced_prompt = enhance_multi_document_prompt(request.query, images_data)
        debug_info.append(f"Enhanced multi-document prompt created for {len(images_data)} pages")
        
        # Group images by document
        doc_image_groups = {}
        for img in images_data:
            doc_name = img['document_name']
            if doc_name not in doc_image_groups:
                doc_image_groups[doc_name] = []
            doc_image_groups[doc_name].append(img)
        
        debug_info.append(f"Processing {len(doc_image_groups)} documents with pages: {', '.join([f'{doc}({len(imgs)}页)' for doc, imgs in doc_image_groups.items()])}")
        
        # Call Ollama VLM
        vlm_response, full_response = await asyncio.to_thread(
            call_ollama_vlm_multipage,
            OLLAMA_URL, MODEL_NAME, enhanced_prompt, images_data, timeout=300
        )
        debug_info.append(f"VLM response received ({len(vlm_response)} chars)")
        
        # Post-process response
        doc_names = list(set(img['document_name'] for img in images_data))
        processed_response = post_process_response(vlm_response, request.query, doc_names)
        debug_info.append("Response post-processed")
        
        # Parse tables if present
        table_data = None
        if '|' in processed_response:
            table_data = table_processor.parse_table_from_text(processed_response)
            if table_data:
                debug_info.append(f"Table detected: {table_data['format']} format")
        
        # Enhanced source information with document breakdown
        source_info = {
            'documents_used': doc_names,
            'pages_used': len(images_data),
            'page_details': [
                {
                    'document_name': img['document_name'],
                    'page_number': img['page_number'],
                    'score': img['score']
                }
                for img in images_data
            ],
            'primary_page': images_data[0]['page_number'] if images_data else 'Unknown',
            'document_breakdown': {
                doc_name: len(imgs) for doc_name, imgs in doc_image_groups.items()
            },
            'coverage_summary': f"{len(doc_image_groups)} documents with {len(images_data)} total pages"
        }
        
        # Get base64 preview
        base64_preview = None
        if images_data and 'base64' in images_data[0]:
            base64_data = images_data[0]['base64']
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',', 1)[1]
            base64_preview = base64_data[:100] + "..." if len(base64_data) > 100 else base64_data
        
        return QueryResponse(
            result=processed_response,
            table_data=table_data,
            source_info=source_info,
            debug_info=debug_info,
            images_info=images_data,
            base64_preview=base64_preview,
            full_response=full_response,
            diagram_base64=diagram_base64_for_frontend
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        debug_info.append(f"Error: {str(e)}")
        
        if request.allow_general_fallback:
            debug_info.append("Falling back to general knowledge")
            return await general_knowledge_fallback(request.query)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Query processing failed: {str(e)}"
            )
        
@app.get("/search-documents", response_model=dict, summary="Search documents by name")
async def search_documents(
    query: str = Query(..., description="Partial or full name to search for"),
    api_key: str = Query(default=API_KEY),
    collection_name: str = Query(DEFAULT_COLLECTION)
):
    """
    Search for documents by name using partial (substring) match, case-insensitive.
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Fetch all documents in the collection
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{COLIVARA_BASE_URL}/documents/",
                headers=headers,
                params={"collection_name": collection_name}
            )

            if response.status_code == 401:
                raise HTTPException(status_code=401, detail="Invalid API key")
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to retrieve documents: {response.text}"
                )

            documents = response.json()

        # Case-insensitive partial match search
        query_lower = query.lower()
        matched_documents = [
            {
                "name": doc.get("name", ""),
                "id": doc.get("id", ""),
                "number_of_pages": doc.get("num_pages", ""),
                "metadata": doc.get("metadata", {})
            }
            for doc in documents
            if query_lower in doc.get("name", "").lower()
        ]

        return {
            "query": query,
            "matched_count": len(matched_documents),
            "matched_documents": matched_documents
        }

    except Exception as e:
        logger.exception("Error searching documents")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Helper endpoint to list documents for debugging
@app.get("/documents", response_model=dict)
async def list_documents(
    api_key: str = Query(default=API_KEY, description="API key for authentication"),
    collection_name: str = Query(DEFAULT_COLLECTION, description="Collection name")
):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            # Fixed parameters
            response = await client.get(
                f"{COLIVARA_BASE_URL}/documents/",
                headers=headers,
                params={"collection_name": collection_name}  # Correct parameter name
            )
            
            if response.status_code == 401:
                raise HTTPException(status_code=401, detail="Invalid API key")
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to list documents: {response.text}"
                )
            
            documents = response.json()
            return {
                "collection_name": collection_name,
                "document_count": len(documents),
                "documents": [
                    {
                        "name": doc.get("name", ""),
                        "id": doc.get("id", ""),
                        "number_of_pages": doc.get("num_pages", ""),
                        "metadata": doc.get("metadata", {})
                    }
                    for doc in documents
                ]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = Form(default=API_KEY),
    collection_name: str = Form(DEFAULT_COLLECTION),
    wait_for_processing: bool = Form(True)
):
    """Upload a single document to ColiVara with comprehensive metadata and duplicate checking"""
    try:
        logger.info(f"Starting upload: {file.filename}")
        
        # Prepare headers for API calls
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Check if document already exists
        existing_docs = await get_existing_documents(headers, collection_name)
        if file.filename in existing_docs:
            logger.info(f"Document {file.filename} already exists in collection {collection_name}")
            # Fetch full metadata for the existing document
            existing_doc = await get_document_info(headers, collection_name, file.filename)
            return UploadResponse(
                message=f"Document already exists in collection '{collection_name}'",
                filename=existing_doc["name"],
                size=existing_doc["metadata"]["file_size"],
                collection_name=collection_name,
                document_id=existing_doc["id"],
                existing_metadata=existing_doc.get("metadata", {})
            )

        # Proceed with upload if document doesn't exist
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
                "source_system": get_system_info(),
                "upload_parameters": {
                    "collection": collection_name,
                    "wait": wait_for_processing
                },
                "file_characteristics": {
                    "extension": os.path.splitext(file.filename)[1]
                }
            }
        }
        
        async with httpx.AsyncClient(timeout=300000.0) as client:
            response = await client.post(
                f"{COLIVARA_BASE_URL}/documents/upsert-document/",
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
            return UploadResponse(
                message="Document uploaded successfully",
                filename=file.filename,
                size=file_size,
                collection_name=collection_name,
                document_id=result.get("id", file.filename)
            )
    
    except HTTPException as he:
        logger.error(f"HTTP error for {file.filename}: {he.detail}")
        raise
    except httpx.TimeoutException:
        logger.error(f"Upload timeout for file: {file.filename}")
        raise HTTPException(status_code=504, detail="Upload timeout - file may be too large")
    except Exception as e:
        logger.exception(f"Upload error for {file.filename}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def get_document_info(headers: dict, collection_name: str, filename: str) -> dict:
    """Return the full document record for `filename`, or None if not found."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{COLIVARA_BASE_URL}/documents/",
            headers=headers,
            params={"collection_name": collection_name}
        )
        resp.raise_for_status()
        for doc in resp.json():
            if doc.get("name") == filename:
                return doc
    return None
   
async def get_existing_documents(headers: dict, collection_name: str) -> set:
    """Retrieve existing documents from collection"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{COLIVARA_BASE_URL}/documents/",
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

async def upload_single_file(
    semaphore: asyncio.Semaphore,
    headers: dict,
    file_path: str,
    filename: str,
    collection_name: str,
    continue_on_error: bool
) -> bool:
    """Upload a single file with comprehensive metadata"""
    async with semaphore:
        try:
            # Read file content
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()
            
            file_size = len(content)
            file_base64 = base64.b64encode(content).decode('utf-8')
            current_time = datetime.now(timezone(timedelta(hours=8)))
            file_extension = os.path.splitext(filename)[1]
            
            # Guess content type if not available
            content_type, _ = mimetypes.guess_type(filename)
            if not content_type:
                content_type = "application/octet-stream"
            
            # Enhanced metadata for bulk upload
            payload = {
                "name": filename,
                "collection_name": collection_name,
                "base64": file_base64,
                "wait": True,
                "metadata": {
                    "file_size": file_size,
                    "content_type": content_type,
                    "original_path": file_path,
                    "file_extension": file_extension,
                    "upload_timestamp": current_time.isoformat(),
                    "processing_time": current_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "upload_method": "bulk",
                    "source_system": get_system_info(),
                    "file_characteristics": {
                        "extension": file_extension,
                        "type_guess": content_type
                    },
                    "processing_context": {
                        "collection": collection_name,
                        "continue_on_error": continue_on_error
                    }
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{COLIVARA_BASE_URL}/documents/upsert-document/",
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                
                if response.status_code not in [200, 201]:
                    error_detail = response.text[:500] if response.text else f"HTTP {response.status_code}"
                    logger.error(f"Failed to upload {filename}: {error_detail}")
                    if not continue_on_error:
                        raise Exception(f"Upload failed: {error_detail}")
                    return False
                    
                return True
                
        except Exception as e:
            logger.error(f"Error uploading {filename}: {str(e)}")
            if not continue_on_error:
                raise
            return False

@app.post("/upload-bulk", response_model=dict,
          summary="Bulk upload from folder",
          description="Upload all files from a folder to ColiVara with comprehensive metadata")
async def upload_bulk(
    folder_path: str = Form(..., description="Path to folder containing documents"),
    api_key: str = Form(default=API_KEY),
    collection_name: str = Form(DEFAULT_COLLECTION),
    max_file_size_mb: int = Form(50),
    continue_on_error: bool = Form(True),
    max_concurrent: int = Form(3)
):
    """Bulk upload documents from a folder with enhanced metadata"""
    try:
        # Validate folder path
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise HTTPException(status_code=400, detail="Invalid folder path")
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Get existing documents to avoid duplicates
        existing_docs = await get_existing_documents(headers, collection_name)
        logger.info(f"Found {len(existing_docs)} existing documents")
        
        # Process files
        new_files = []
        max_size_bytes = max_file_size_mb * 1024 * 1024
        skipped_files = {"existing": [], "too_large": [], "invalid": []}
        
        # Supported file extensions
        supported_extensions = {'.pdf', '.doc', '.docx', '.txt', '.png', '.jpg', 
                               '.jpeg', '.tiff', '.bmp', '.csv', '.xls', '.xlsx',
                               '.ppt', '.pptx', '.md', '.html', '.json'}
        
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
            task = upload_single_file(
                semaphore, headers, file_path, filename, 
                collection_name, continue_on_error
            )
            upload_tasks.append(task)
        
        # Wait for all uploads to complete
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
                failed_files.append({"filename": filename, "error": "Upload failed without exception"})
        
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
    
# Helper endpoint to check upload status
@app.get("/upload-status/{document_name}")
async def check_upload_status(
    document_name: str,
    api_key: str = API_KEY,
    collection_name: str = DEFAULT_COLLECTION
):
    """Improved upload status check with comprehensive error handling"""
    
    logger.info(f"Checking upload status for: {document_name}")
    
    try:
        # Decode the document name
        decoded_document_name = urllib.parse.unquote(document_name)
        logger.info(f"Decoded document name: {decoded_document_name}")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Step 1: Try to list documents
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Listing documents in collection: {collection_name}")
            
            response = await client.get(
                f"{COLIVARA_BASE_URL}/documents/",
                headers=headers,
                params={"collection_name": collection_name}
            )
            
            logger.info(f"List documents response status: {response.status_code}")
            
            if response.status_code == 401:
                return {"status": "unauthorized", "detail": "Invalid API key"}
            elif response.status_code == 404:
                return {"status": "collection_not_found", "detail": f"Collection '{collection_name}' not found"}
            elif response.status_code != 200:
                error_detail = response.text[:500]  # Limit error message length
                logger.error(f"API error: {error_detail}")
                return {"status": "api_error", "detail": f"API returned {response.status_code}: {error_detail}"}
            
            documents = response.json()
            logger.info(f"Found {len(documents)} documents in collection")
            
            # Step 2: Search for the document
            found_document = None
            for doc in documents:
                doc_name = doc.get("name", "")
                if doc_name == document_name or doc_name == decoded_document_name:
                    found_document = doc
                    logger.info(f"Found document: {doc_name}")
                    break
            
            if found_document:
                return {
                    "status": "processed",
                    "document": {
                        "name": found_document.get("name"),
                        "id": found_document.get("id"),
                        "num_pages": found_document.get("num_pages", 0),
                        "metadata": found_document.get("metadata", {}),
                        "collection": collection_name
                    },
                    "found_in_collection": True
                }
            else:
                # List available documents for debugging
                available_docs = [doc.get("name", "") for doc in documents[:10]]  # First 10 docs
                logger.warning(f"Document not found. Available documents: {available_docs}")
                
                return {
                    "status": "not_found",
                    "detail": f"Document '{decoded_document_name}' not found in collection '{collection_name}'",
                    "searched_for": [document_name, decoded_document_name],
                    "available_documents_sample": available_docs,
                    "total_documents_in_collection": len(documents)
                }
                
    except httpx.TimeoutException:
        logger.error("Request timed out")
        return {"status": "timeout", "detail": "Request timed out - ColiVara may be slow"}
    except httpx.RequestError as e:
        logger.error(f"Network error: {str(e)}")
        return {"status": "network_error", "detail": f"Failed to connect to ColiVara: {str(e)}"}
    except Exception as e:
        logger.exception("Unexpected error in upload status check")
        return {"status": "error", "detail": f"Unexpected error: {str(e)}"}

@app.delete("/delete/{filename}", response_model=DeleteResponse)
async def delete_document(
    filename: str = Path(..., description="Document filename to delete"),
    api_key: str = Query(default=API_KEY, description="API key for authentication"),
    collection_name: str = Query(DEFAULT_COLLECTION, description="Collection name")
):
    try:
        decoded_filename = urllib.parse.unquote(filename)
        logger.info(f"Deleting document: '{decoded_filename}'")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Use URL-safe encoding
        encoded_filename = urllib.parse.quote(decoded_filename, safe='')
        
        async with httpx.AsyncClient() as client:
            # Fixed endpoint URL and parameters
            delete_response = await client.delete(
                f"{COLIVARA_BASE_URL}/documents/delete-document/{encoded_filename}/",  # Added trailing slash
                headers=headers,
                params={"collection_name": collection_name}
            )
            
            if delete_response.status_code == 204:
                return DeleteResponse(
                    message=f"Document '{decoded_filename}' deleted successfully",
                    filename=decoded_filename
                )
            elif delete_response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document '{decoded_filename}' not found in ColiVara collection '{collection_name}'"
                )
            else:
                raise HTTPException(
                    status_code=delete_response.status_code,
                    detail=f"Delete failed: {delete_response.text}"
                )
            
    except HTTPException:
        raise
    except httpx.RequestError as e:
        logger.error(f"Network error when deleting document '{filename}': {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to ColiVara service: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting document '{filename}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.delete("/delete-matching", response_model=dict)
async def delete_documents_matching(
    keyword: str = Query(..., description="Substring to match in document names (e.g., '_1')"),
    api_key: str = Query(default=API_KEY, description="API key for authentication"),
    collection_name: str = Query(DEFAULT_COLLECTION, description="Collection name")
):
    """
    Delete all documents that contain the given substring in their filename.
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{COLIVARA_BASE_URL}/documents/",
                headers=headers,
                params={"collection_name": collection_name}
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to list documents: {response.text}"
                )

            documents = response.json()

        # Find documents matching the keyword (case-insensitive)
        keyword_lower = keyword.lower()
        matching_docs = [
            doc for doc in documents
            if keyword_lower in doc.get("name", "").lower()
        ]

        if not matching_docs:
            return {
                "message": f"No documents found matching '{keyword}'",
                "matched_count": 0,
                "deleted": []
            }

        # Delete them one by one
        deleted = []
        failed = []

        async with httpx.AsyncClient() as client:
            for doc in matching_docs:
                filename = doc.get("name", "")
                encoded_filename = urllib.parse.quote(filename, safe='')
                delete_url = f"{COLIVARA_BASE_URL}/documents/delete-document/{encoded_filename}/"

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
        logger.exception("Bulk delete by match failed")
        raise HTTPException(status_code=500, detail=f"Bulk delete failed: {str(e)}")

# Add this new endpoint to your existing API endpoints section

@app.get("/preview/{document_name}", response_model=dict,
         summary="Preview document pages", 
         description="Get document preview by searching for specific document pages")
async def preview_document(
    document_name: str = Path(..., description="Document name to preview"),
    page_number: int = Query(default=None, ge=1, description="Specific page number (optional)"),
    max_pages: int = Query(default=1, ge=1, le=10, description="Maximum number of pages to return"),
    api_key: str = Query(default=API_KEY, description="API key for authentication"),
    collection_name: str = Query(DEFAULT_COLLECTION, description="Collection name")
):
    """
    Preview document pages using ColiVara search.
    Uses the same search mechanism as your query endpoint to find document pages.
    """
    try:
        # Enhanced document name handling for alphanumeric characters
        decoded_document_name = urllib.parse.unquote(document_name)
        # Additional cleaning for special characters that might cause issues
        clean_document_name = decoded_document_name.strip()
        
        logger.info(f"Previewing document: '{clean_document_name}' (original: '{document_name}')")
        
        # Step 1: First verify the document exists using your existing pattern
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Check if document exists using your existing function
        existing_docs = await get_existing_documents(headers, collection_name)
        
        # Enhanced document matching - try multiple variations
        document_found = False
        matched_document_name = None
        
        # Direct match first
        if clean_document_name in existing_docs:
            document_found = True
            matched_document_name = clean_document_name
        else:
            # Try case-insensitive matching for alphanumeric issues
            for existing_doc in existing_docs:
                if existing_doc.lower() == clean_document_name.lower():
                    document_found = True
                    matched_document_name = existing_doc
                    break
            
            # If still not found, try partial matching for documents with special characters
            if not document_found:
                for existing_doc in existing_docs:
                    # Remove common separators and compare
                    clean_existing = existing_doc.replace('_', '').replace('-', '').replace(' ', '').lower()
                    clean_target = clean_document_name.replace('_', '').replace('-', '').replace(' ', '').lower()
                    if clean_existing == clean_target:
                        document_found = True
                        matched_document_name = existing_doc
                        break
        
        if not document_found:
            logger.warning(f"Document '{clean_document_name}' not found. Available docs: {list(existing_docs)[:5]}")
            raise HTTPException(
                status_code=404,
                detail=f"Document '{clean_document_name}' not found in collection '{collection_name}'"
            )
        
        logger.info(f"Found matching document: '{matched_document_name}'")
        
        # Step 2: Use ColiVara search to find pages from this document
        # Following the same pattern as your query endpoint
        rag_client = ColiVara(api_key=api_key, base_url=COLIVARA_BASE_URL_QUERY)
        
        # Enhanced search strategy - use document name parts as search terms
        search_terms = []
        
        # Add the document name itself as a search term
        search_terms.append(matched_document_name)
        
        # Add individual words from document name for better matching
        doc_words = matched_document_name.replace('_', ' ').replace('-', ' ').split()
        search_terms.extend([word for word in doc_words if len(word) > 2])
        
        # Add generic terms
        search_terms.extend(["document", "content", "page", "text"])
        
        # Create search query
        search_query = " ".join(search_terms[:5])  # Limit to avoid too long queries
        
        logger.info(f"Using search query: '{search_query}'")
        
        # Search with higher top_k to get more pages, following your query pattern
        search_results = await asyncio.to_thread(
            rag_client.search,
            query=search_query,
            collection_name=collection_name,
            top_k=100  # Higher number to get more pages for filtering
        )
        
        if not search_results.results:
            logger.warning("No search results returned")
            raise HTTPException(
                status_code=404,
                detail="No content found for document preview"
            )
        
        # Step 3: Enhanced filtering for the target document
        document_pages = []
        for doc in search_results.results:
            doc_name = getattr(doc, 'document_name', '')
            
            # Multiple matching strategies for robust document name matching
            if (doc_name == matched_document_name or 
                doc_name == clean_document_name or
                doc_name.lower() == matched_document_name.lower() or
                doc_name.lower() == clean_document_name.lower()):
                document_pages.append(doc)
                continue
            
            # Try normalized matching for alphanumeric issues
            normalized_doc_name = doc_name.replace('_', '').replace('-', '').replace(' ', '').lower()
            normalized_target = matched_document_name.replace('_', '').replace('-', '').replace(' ', '').lower()
            
            if normalized_doc_name == normalized_target:
                document_pages.append(doc)
        
        logger.info(f"Found {len(document_pages)} pages for document '{matched_document_name}'")
        
        if not document_pages:
            logger.warning(f"No pages found after filtering. Search returned {len(search_results.results)} results")
            # Debug: log some document names from search results
            sample_docs = [getattr(doc, 'document_name', 'N/A') for doc in search_results.results[:5]]
            logger.info(f"Sample document names from search: {sample_docs}")
            
            raise HTTPException(
                status_code=404,
                detail=f"No pages found for document '{matched_document_name}'"
            )
        
        # Step 4: Sort by page number and filter by requested page if specified
        # Enhanced sorting with better error handling
        def safe_get_page_number(doc):
            try:
                return getattr(doc, 'page_number', 0) or 0
            except:
                return 0
        
        document_pages.sort(key=safe_get_page_number)
        
        if page_number is not None:
            # Filter for specific page and surrounding pages
            target_pages = [doc for doc in document_pages 
                          if safe_get_page_number(doc) >= page_number]
        else:
            target_pages = document_pages
        
        # Limit to max_pages
        target_pages = target_pages[:max_pages]
        
        if not target_pages:
            available_pages = [safe_get_page_number(doc) for doc in document_pages]
            raise HTTPException(
                status_code=404,
                detail=f"No pages found starting from page {page_number}. Available pages: {sorted(set(available_pages))}" if page_number else "No pages found"
            )
        
        # Step 5: Process images using your existing function
        # This follows the same pattern as your query endpoint
        try:
            images_data = combine_images_for_vlm(target_pages)
        except Exception as e:
            logger.warning(f"Image processing failed: {str(e)}")
            images_data = None
        
        if not images_data:
            # Fallback to text-only response, following your query pattern
            logger.info("No valid image data found - using text-only response")
            
            preview_pages = []
            for doc in target_pages:
                page_data = {
                    "page_number": safe_get_page_number(doc),
                    "document_name": getattr(doc, 'document_name', ''),
                    "text_content": getattr(doc, 'text', ''),
                    "score": getattr(doc, 'normalized_score', 0),
                    "has_image": False
                }
                preview_pages.append(page_data)
            
            return {
                "document_name": matched_document_name,
                "collection_name": collection_name,
                "pages_found": len(preview_pages),
                "preview_mode": "text_only",
                "pages": preview_pages,
                "success": True
            }
        
        # Step 6: Return preview with images (following your existing image processing)
        preview_pages = []
        for img_data in images_data:
            page_data = {
                "page_number": img_data.get('page_number', 0),
                "document_name": img_data.get('document_name', ''),
                "score": img_data.get('score', 0),
                "has_image": True,
                "image_base64": img_data.get('base64', ''),
                "text_content": img_data.get('text', '')
            }
            
            # Add image preview (first 100 chars for reference, following your pattern)
            if img_data.get('base64'):
                base64_data = img_data['base64']
                if base64_data.startswith('data:'):
                    base64_data = base64_data.split(',', 1)[1]
                page_data["image_preview"] = base64_data[:100] + "..." if len(base64_data) > 100 else base64_data
            
            preview_pages.append(page_data)
        
        return {
            "document_name": matched_document_name,
            "collection_name": collection_name,
            "pages_found": len(preview_pages),
            "preview_mode": "with_images",
            "requested_page": page_number,
            "max_pages": max_pages,
            "pages": preview_pages,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document preview error for '{document_name}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Preview failed: {str(e)}"
        )

# Simple helper endpoint to check if document exists and get basic info
@app.get("/document-exists/{document_name}", response_model=dict,
         summary="Check if document exists",
         description="Check if document exists in collection and get basic metadata")
async def check_document_exists(
    document_name: str = Path(..., description="Document name to check"),
    api_key: str = Query(default=API_KEY, description="API key for authentication"),
    collection_name: str = Query(DEFAULT_COLLECTION, description="Collection name")
):
    """
    Check if document exists using your existing get_existing_documents function.
    Enhanced with better alphanumeric character handling.
    """
    try:
        decoded_document_name = urllib.parse.unquote(document_name)
        clean_document_name = decoded_document_name.strip()
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Use your existing function to check for document existence
        existing_docs = await get_existing_documents(headers, collection_name)
        
        # Enhanced matching logic
        exists = False
        matched_name = None
        
        # Direct match
        if clean_document_name in existing_docs:
            exists = True
            matched_name = clean_document_name
        else:
            # Case-insensitive matching
            for existing_doc in existing_docs:
                if existing_doc.lower() == clean_document_name.lower():
                    exists = True
                    matched_name = existing_doc
                    break
            
            # Normalized matching for alphanumeric issues
            if not exists:
                for existing_doc in existing_docs:
                    clean_existing = existing_doc.replace('_', '').replace('-', '').replace(' ', '').lower()
                    clean_target = clean_document_name.replace('_', '').replace('-', '').replace(' ', '').lower()
                    if clean_existing == clean_target:
                        exists = True
                        matched_name = existing_doc
                        break
        
        return {
            "document_name": clean_document_name,
            "matched_document_name": matched_name,
            "collection_name": collection_name,
            "exists": exists,
            "total_documents_in_collection": len(existing_docs),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Document existence check error for '{document_name}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check document existence: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test connection to ColiVara
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{COLIVARA_BASE_URL}/")
            colivara_status = "healthy" if response.status_code < 500 else "unhealthy"
    except:
        colivara_status = "unreachable"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),  # You can use datetime.now().isoformat()
        "services": {
            "api": "healthy",
            "colivara": colivara_status,
        },
        "version": "2.0.0"
    }

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document Q&A API with ColiVara Integration",
        "version": "2.0.0",
        "endpoints": {
            "query": "/query (POST) - Search across documents using a question",
            "upload": "/upload (POST) - Upload single document",
            "upload_bulk": "/upload-bulk (POST) - Bulk upload from folder",
            "search_matching": "/search-matching (GET) - Search documents by partial name match",
            "list_documents": "/documents (GET) - List all documents in collection",
            "delete": "/delete/{filename} (DELETE) - Delete document by exact filename",
            "delete_matching": "/delete-matching (DELETE) - Bulk delete by matching keyword in filename",
            
            "health": "/health (GET) - API health check"
        },
        "authentication": {
            "methods": ["Query parameter (api_key)", "Bearer token (Authorization header)"],
            "note": "All endpoints require valid API key"
        },
        "configurations": {
            "colivara_base_url": COLIVARA_BASE_URL,
            "minio_bucket": MINIO_BUCKET,
            "default_collection": DEFAULT_COLLECTION,
            "ollama_url": OLLAMA_URL,
            "model_name": MODEL_NAME
        },
        "supported_file_types": [".pdf", ".doc", ".docx", ".txt"],
        "examples": {
            "list_documents": f"GET /documents?api_key=YOUR_KEY&collection_name=default_collection",
            "delete_document": f"DELETE /delete/document.pdf?api_key=YOUR_KEY&collection_name=default_collection",
            "delete_matching": f"DELETE /delete-matching?keyword=_1&api_key=YOUR_KEY&collection_name=default_collection",
            "search_matching": f"GET /search-matching?query=invoice&api_key=YOUR_KEY&collection_name=default_collection",
            "upload_document": f"POST /upload with form data: file, api_key, collection_name"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)