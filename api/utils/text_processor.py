"""
Text Processing Utilities Module

Contains functions for:
- Query preprocessing and normalization
- Legal reference normalization
- Chinese-Arabic number conversion
- Mathematical expression handling
- Language detection
- Section context extraction
"""

import re
from typing import List, Dict, Union, Tuple, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

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

# Section cleanup pattern
SECTION_CLEANUP_PATTERN = re.compile(
    r'(第[零一二三四五六七八九十百千万亿\d]+\s*[条款项目节章部分段篇卷编表图例式附件附录])[：:、．.，,]?\s+([零一二三四五六七八九十百千万亿\d]+)(?=\s|$)'
)

# Table and formatting patterns
TABLE_CLEANUP_PATTERN = re.compile(r'([—\-]{3,})')
DUPLICATE_HEADER_PATTERN = re.compile(r'(\*\*[^*]+\*\*)\s*\1')

# Comprehensive Chinese to Arabic mapping
CHINESE_TO_ARABIC_COMPREHENSIVE = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, 
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '百': 100, '千': 1000, '万': 10000, '亿': 100000000
}

# Arabic to Chinese mapping
ARABIC_TO_CHINESE = {
    0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五',
    6: '六', 7: '七', 8: '八', 9: '九', 10: '十'
}

# Math conversions mapping
MATH_CONVERSIONS = {
    '加': '+', '减': '-', '乘以': '×', '乘': '×', '除以': '÷', '除': '÷',
    '等于': '=', '大于': '>', '小于': '<', '不少于': '>=', '不超过': '<=',
    '至少': '>=', '最少': '>=', '最多': '<=', '超过': '>', '低于': '<', '高于': '>',
}


@lru_cache(maxsize=500)
def detect_query_language(query: str) -> str:
    """
    Detect primary language of query - cached
    
    Args:
        query: Query string
        
    Returns:
        'chinese' or 'english'
    """
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
    english_chars = len(re.findall(r'[a-zA-Z]', query))
    return 'chinese' if chinese_chars > english_chars else 'english'


def preprocess_query(query: str, query_processor=None) -> str:
    """
    Preserve exact query structure with minimal normalization
    
    Args:
        query: Original query string
        query_processor: Optional ModernQueryProcessor instance
        
    Returns:
        Normalized query
    """
    if query_processor:
        try:
            features = query_processor.extract_semantic_features(query)
            return features.get('normalized_query', query)
        except Exception as e:
            logger.warning(f"Query processor preprocessing failed: {e}")
    
    # Fallback: basic normalization
    # Remove extra whitespace
    normalized = WHITESPACE_PATTERN.sub(' ', query)
    normalized = normalized.strip()
    
    return normalized


def expand_query_terms(query: str, query_processor=None) -> str:
    """
    Controlled expansion with exact match priority
    
    Args:
        query: Query to expand
        query_processor: Optional ModernQueryProcessor instance
        
    Returns:
        Expanded query string
    """
    if query_processor:
        try:
            return query_processor.create_prioritized_search_query(query)
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
    
    # Fallback: return original
    return query


@lru_cache(maxsize=100)
def convert_chinese_to_arabic(chinese_num: str) -> int:
    """
    Enhanced Chinese to Arabic conversion with complete coverage for all section numbers
    
    Args:
        chinese_num: Chinese numeral string
        
    Returns:
        Arabic number (int)
    """
    if not chinese_num or chinese_num.isdigit():
        return int(chinese_num) if chinese_num.isdigit() else 0
    
    chinese_num = chinese_num.strip()
    
    # Complete mappings for all numbers 0-99
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
        
        # Common hundreds
        '一百': 100, '二百': 200, '三百': 300, '四百': 400, '五百': 500,
        '六百': 600, '七百': 700, '八百': 800, '九百': 900, '一千': 1000
    }
    
    # Check simple mappings first
    if chinese_num in simple_mappings:
        return simple_mappings[chinese_num]
    
    # Handle 十X patterns
    if chinese_num.startswith('十'):
        if len(chinese_num) == 1:
            return 10
        remainder = chinese_num[1:]
        if len(remainder) == 1 and remainder in simple_mappings and simple_mappings[remainder] < 10:
            return 10 + simple_mappings[remainder]
    
    # Handle X十Y patterns
    tens_pattern = re.match(r'^([一二三四五六七八九])十([一二三四五六七八九]?)$', chinese_num)
    if tens_pattern:
        tens_digit = simple_mappings.get(tens_pattern.group(1), 0)
        ones_digit = simple_mappings.get(tens_pattern.group(2), 0) if tens_pattern.group(2) else 0
        return tens_digit * 10 + ones_digit
    
    # Complex number parsing
    try:
        result = 0
        current = 0
        temp_num = 0
        
        for char in chinese_num:
            if char not in CHINESE_TO_ARABIC_COMPREHENSIVE:
                continue
                
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
        
        result += current + temp_num
        return max(0, result)
        
    except Exception as e:
        logger.error(f"Error in complex parsing for '{chinese_num}': {e}")
        # Fallback: extract digits
        digits = ''.join(c for c in chinese_num if c.isdigit())
        return int(digits) if digits else 0


@lru_cache(maxsize=100)
def convert_arabic_to_chinese_numeral(number: int) -> str:
    """
    Convert Arabic numerals to Chinese legal numerals with complete coverage
    
    Args:
        number: Arabic number (int)
        
    Returns:
        Chinese numeral string
    """
    if number == 0:
        return "零"
    
    if number < 0:
        return "负" + convert_arabic_to_chinese_numeral(-number)
    
    # Complete mapping for 1-20
    direct_mappings = {
        1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
        11: "十一", 12: "十二", 13: "十三", 14: "十四", 15: "十五",
        16: "十六", 17: "十七", 18: "十八", 19: "十九", 20: "二十"
    }
    
    if number in direct_mappings:
        return direct_mappings[number]
    
    # Handle larger numbers
    def convert_section(num, unit_name=""):
        if num == 0:
            return ""
        if num < 10:
            return direct_mappings.get(num, str(num)) + unit_name
        elif num < 21:
            return direct_mappings.get(num, f"十{direct_mappings.get(num-10, str(num-10))}") + unit_name
        elif num < 100:
            tens = num // 10
            ones = num % 10
            result = direct_mappings[tens] + "十"
            if ones > 0:
                result += direct_mappings[ones]
            return result + unit_name
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
                        result += "一十"
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
            return str(num) + unit_name
    
    # Process the number in sections
    if number < 10000:
        return convert_section(number)
    elif number < 100000000:
        wan_part = number // 10000
        remainder = number % 10000
        result = convert_section(wan_part) + "万"
        if remainder > 0:
            if remainder < 1000:
                result += "零"
            result += convert_section(remainder)
        return result
    else:
        yi_part = number // 100000000
        remainder = number % 100000000
        result = convert_section(yi_part) + "亿"
        if remainder > 0:
            if remainder < 10000000:
                result += "零"
            result += convert_arabic_to_chinese_numeral(remainder)
        return result


def normalize_legal_references(text: str) -> str:
    """
    Normalize legal references - ONLY convert Arabic to Chinese, preserve Chinese as-is
    
    Args:
        text: Text containing legal references
        
    Returns:
        Text with normalized references
    """
    # Identify existing Chinese references to preserve them
    chinese_references = set()
    chinese_section_pattern = re.compile(
        r'第\s*([零一二三四五六七八九十百千万亿]+)\s*([条款项目节章部分段篇卷编表图例式附件附录])',
        re.UNICODE
    )
    
    for match in chinese_section_pattern.finditer(text):
        chinese_references.add((match.start(), match.end()))
    
    def arabic_to_chinese_conversion(match):
        """Convert Arabic numerals to Chinese only if not overlapping"""
        match_start, match_end = match.span()
        
        # Check overlap with Chinese references
        for chinese_start, chinese_end in chinese_references:
            if (match_start < chinese_end and match_end > chinese_start):
                return match.group(0)
        
        # Convert Arabic to Chinese
        prefix = match.group(1) if match.group(1) else ""
        section_marker = match.group(2) if match.group(2) else "第"
        num_str = match.group(3)
        section_type = match.group(4)
        
        try:
            if num_str.isdigit():
                num = int(num_str)
                chinese_num = convert_arabic_to_chinese_numeral(num)
                return f"{prefix}{section_marker}{chinese_num}{section_type}"
            return match.group(0)
        except (ValueError, TypeError):
            return match.group(0)
    
    # Pattern for Arabic section references
    arabic_section_pattern = re.compile(
        r'(\s*)(第)?\s*(\d+)\s*([条款项目节章部分段篇卷编表图例式附件附录])',
        re.UNICODE
    )
    
    # Convert Arabic references
    normalized_text = arabic_section_pattern.sub(arabic_to_chinese_conversion, text)
    
    # Normalize hierarchy spacing
    hierarchy_pattern = re.compile(
        r'(第[零一二三四五六七八九十百千万亿\d]+[条款项目节章部分段篇卷编])(\s+)(第[零一二三四五六七八九十百千万亿\d]+[条款项目节章部分段篇卷编])',
        re.UNICODE
    )
    normalized_text = hierarchy_pattern.sub(r'\1 \3', normalized_text)
    
    # Clean up section formatting
    normalized_text = SECTION_CLEANUP_PATTERN.sub(r'\1', normalized_text)
    
    return normalized_text


def normalize_numbers_fast(text: str) -> str:
    """
    Enhanced number normalization - converts Arabic section numbers to Chinese
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert Arabic section references to Chinese
    text = normalize_legal_references(text)
    
    # Handle ranges
    def convert_range_to_chinese(match):
        start_num = match.group(1)
        end_num = match.group(2)
        
        try:
            if start_num.isdigit() and end_num.isdigit():
                start_chinese = convert_arabic_to_chinese_numeral(int(start_num))
                end_chinese = convert_arabic_to_chinese_numeral(int(end_num))
                return f"{start_chinese}到{end_chinese}范围 {start_num}-{end_num}"
            return match.group(0)
        except:
            return match.group(0)
    
    # Convert numeric ranges
    text = re.sub(r'(\d+)[-~至到](\d+)', convert_range_to_chinese, text)
    
    # Handle percentages
    text = PERCENTAGE_CONVERSION_PATTERN.sub(r'\1%', text)
    text = PERCENTAGE_EXPANSION_PATTERN.sub(r'\1 percent \1% 百分之\1', text)
    
    return text


def normalize_mathematical_expressions(text: str) -> str:
    """
    Convert Chinese mathematical terms to standardized forms
    
    Args:
        text: Text containing math expressions
        
    Returns:
        Text with normalized math expressions
    """
    for chinese, symbol in MATH_CONVERSIONS.items():
        text = re.sub(f'\\b{chinese}\\b', symbol, text)
    
    return text


def extract_mathematical_expressions(text: str) -> List[Dict]:
    """
    Extract and identify mathematical expressions from text
    
    Args:
        text: Text to analyze
        
    Returns:
        List of dicts with match info
    """
    expressions = []
    
    patterns = [ARITHMETIC_PATTERN, PERCENTAGE_PATTERN, FRACTION_PATTERN, RANGE_PATTERN, CHINESE_MATH_PATTERN]
    for pattern in patterns:
        for match in pattern.finditer(text):
            expressions.append({
                'match': match.group(),
                'start': match.start(),
                'end': match.end(),
                'type': 'mathematical'
            })
    
    return expressions


def perform_basic_calculations(expression: str) -> Union[float, str]:
    """
    Safely evaluate basic mathematical expressions
    
    Args:
        expression: Math expression string
        
    Returns:
        Calculation result or original string if invalid
    """
    try:
        # Normalize and clean
        expression = normalize_mathematical_expressions(expression)
        expression = expression.replace('×', '*').replace('÷', '/')
        
        # Only allow safe characters
        if not all(c in '0123456789+-*/.() ' for c in expression):
            return expression
        
        # Evaluate
        result = eval(expression)
        return round(result, 4) if isinstance(result, float) else result
        
    except Exception:
        return expression


@lru_cache(maxsize=500)
def get_section_context(text: str, position: int, window_size: int = 200) -> str:
    """
    Get context around a section marker - cached
    
    Args:
        text: Full text
        position: Position of section marker
        window_size: Context window size
        
    Returns:
        Context substring
    """
    start = max(0, position - window_size)
    end = min(len(text), position + window_size)
    return text[start:end]


def preserve_original_structure(text: str, table_processor=None) -> str:
    """
    Enhanced structure preservation with better table formatting
    
    Args:
        text: Text to preserve structure for
        table_processor: Optional TableProcessor instance
        
    Returns:
        Formatted text
    """
    # Normalize lists
    text = NUMBERED_LIST_PATTERN.sub(r'\1\2 ', text)
    text = BULLET_PATTERN.sub(r'\1\2 ', text)
    
    # Convert tables if processor available
    if table_processor:
        try:
            text = table_processor.convert_text_to_markdown_table(text)
        except Exception as e:
            logger.warning(f"Table conversion failed: {e}")
    
    # Clean up formatting
    text = TABLE_CLEANUP_PATTERN.sub('---', text)
    text = DUPLICATE_HEADER_PATTERN.sub(r'\1', text)
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = MULTIPLE_NEWLINES_PATTERN.sub('\n\n', text)
    
    return text.strip()
