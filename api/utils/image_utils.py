"""
Image Utilities Module

Contains functions for image processing and validation:
- Base64 image validation and cleaning
- Image combination for VLM processing
- Format detection and size checks
"""

import base64
import io
from typing import List, Dict, Tuple
from functools import lru_cache
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=200)
def validate_base64_image(encoded_blob: str) -> Tuple[str, int, str, Tuple[int, int]]:
    """
    Validate and clean base64 image data - cached for performance
    
    Args:
        encoded_blob: Base64 encoded image data (with or without data URI prefix)
        
    Returns:
        Tuple of (clean_base64, size_in_bytes, format, dimensions)
        
    Raises:
        ValueError: If image data is invalid
    """
    try:
        # Remove data URI prefix if present
        if encoded_blob.startswith('data:'):
            encoded_blob = encoded_blob.split(',', 1)[1]
        
        # Decode and validate
        image_bytes = base64.b64decode(encoded_blob)
        image = Image.open(io.BytesIO(image_bytes))
        
        return encoded_blob, len(image_bytes), image.format, image.size
        
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")


def combine_images_for_vlm(page_docs) -> List[Dict]:
    """
    Prepare multiple page images for VLM processing - highly optimized
    
    Args:
        page_docs: List of document objects with img_base64 attribute
        
    Returns:
        List of dicts with image data, metadata, and scores
    """
    combined_images = []
    
    for doc in page_docs:
        # Skip documents without images
        if not hasattr(doc, 'img_base64') or not doc.img_base64:
            continue
        
        # Basic data extraction
        base64_data = doc.img_base64
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]
        
        # Quick format detection from first few bytes (no full decode for speed)
        try:
            decoded_start = base64.b64decode(base64_data[:20])
            if decoded_start.startswith(b'\xff\xd8'):
                image_format = 'JPEG'
            elif decoded_start.startswith(b'\x89PNG'):
                image_format = 'PNG'
            else:
                image_format = 'unknown'
        except Exception:
            image_format = 'unknown'
        
        combined_images.append({
            'base64': base64_data,
            'page_number': getattr(doc, 'page_number', 'unknown'),
            'document_name': getattr(doc, 'document_name', 'unknown'),
            'format': image_format,
            'score': getattr(doc, 'normalized_score', 0)
        })
    
    return combined_images


def get_image_info(base64_data: str) -> Dict:
    """
    Get detailed information about a base64 encoded image
    
    Args:
        base64_data: Base64 encoded image
        
    Returns:
        Dict with format, size, dimensions, and mode info
    """
    try:
        # Clean data
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]
        
        # Decode and analyze
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        return {
            'format': image.format,
            'mode': image.mode,
            'size_bytes': len(image_bytes),
            'width': image.size[0],
            'height': image.size[1],
            'aspect_ratio': round(image.size[0] / image.size[1], 2) if image.size[1] > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to get image info: {e}")
        return {}


def resize_image_if_needed(image: Image.Image, max_size: Tuple[int, int] = (1920, 1080)) -> Image.Image:
    """
    Resize image if it exceeds max dimensions while preserving aspect ratio
    
    Args:
        image: PIL Image object
        max_size: Maximum (width, height) tuple
        
    Returns:
        Resized image or original if within limits
    """
    if image.size[0] <= max_size[0] and image.size[1] <= max_size[1]:
        return image
    
    # Calculate new size preserving aspect ratio
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def convert_image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """
    Convert PIL Image to base64 string
    
    Args:
        image: PIL Image object
        format: Output format (PNG, JPEG, etc.)
        
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')
