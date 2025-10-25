"""
Table processing module for detecting and parsing tables in text.
Handles various table formats: pipe-separated, colon-separated, space-separated, and CSV.
"""
import re
import io
import pandas as pd
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
from tabulate import tabulate

logger = logging.getLogger(__name__)


class TablePatterns:
    """Compiled regex patterns for table detection and parsing"""
    
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


# Global patterns instance
_PATTERNS = TablePatterns()


@dataclass
class TableDetectionConfig:
    """Configuration for table detection thresholds"""
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
        Convert any text table to markdown with better structure preservation
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
