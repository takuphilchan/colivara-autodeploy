"""
Validation utilities.
Provides validation functions for user input and data.
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email:
        return False, "Email is required"
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Invalid email format"
    
    if len(email) > 254:  # RFC 5321
        return False, "Email too long"
    
    return True, ""


def validate_username(username: str, min_length: int = 3, max_length: int = 50) -> Tuple[bool, str]:
    """
    Validate username format.
    
    Args:
        username: Username to validate
        min_length: Minimum length
        max_length: Maximum length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not username:
        return False, "Username is required"
    
    if len(username) < min_length:
        return False, f"Username must be at least {min_length} characters"
    
    if len(username) > max_length:
        return False, f"Username must be at most {max_length} characters"
    
    # Allow alphanumeric, underscore, and hyphen
    pattern = r'^[a-zA-Z0-9_-]+$'
    if not re.match(pattern, username):
        return False, "Username can only contain letters, numbers, underscores, and hyphens"
    
    # Must start with letter or number
    if not username[0].isalnum():
        return False, "Username must start with a letter or number"
    
    return True, ""


def validate_password(password: str, min_length: int = 8, max_length: int = 128) -> Tuple[bool, str]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        min_length: Minimum length
        max_length: Maximum length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not password:
        return False, "Password is required"
    
    if len(password) < min_length:
        return False, f"Password must be at least {min_length} characters"
    
    if len(password) > max_length:
        return False, f"Password must be at most {max_length} characters"
    
    # Check for at least one number (optional, can be enabled)
    # if not re.search(r'\d', password):
    #     return False, "Password must contain at least one number"
    
    # Check for at least one uppercase letter (optional)
    # if not re.search(r'[A-Z]', password):
    #     return False, "Password must contain at least one uppercase letter"
    
    # Check for at least one lowercase letter (optional)
    # if not re.search(r'[a-z]', password):
    #     return False, "Password must contain at least one lowercase letter"
    
    return True, ""


def validate_password_match(password: str, confirm_password: str) -> Tuple[bool, str]:
    """
    Validate that passwords match.
    
    Args:
        password: First password
        confirm_password: Confirmation password
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if password != confirm_password:
        return False, "Passwords do not match"
    
    return True, ""


def sanitize_input(text: str, max_length: int = None) -> str:
    """
    Sanitize user input by removing dangerous characters.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Limit length if specified
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


def validate_json_structure(data: dict, required_fields: list) -> Tuple[bool, str]:
    """
    Validate that JSON data contains required fields.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Data must be a dictionary"
    
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, ""


def validate_file_extension(filename: str, allowed_extensions: set) -> Tuple[bool, str]:
    """
    Validate file extension.
    
    Args:
        filename: Name of file
        allowed_extensions: Set of allowed extensions
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not filename or '.' not in filename:
        return False, "Invalid filename"
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if ext not in allowed_extensions:
        return False, f"File type .{ext} not allowed. Allowed types: {', '.join(allowed_extensions)}"
    
    return True, ""


def is_safe_path(path: str, base_path: str) -> bool:
    """
    Check if a path is safe (no directory traversal).
    
    Args:
        path: Path to check
        base_path: Base directory path
        
    Returns:
        True if path is safe, False otherwise
    """
    import os
    
    # Resolve absolute paths
    abs_path = os.path.abspath(path)
    abs_base = os.path.abspath(base_path)
    
    # Check if the path is within the base directory
    return abs_path.startswith(abs_base)


def validate_api_key_format(api_key: str) -> Tuple[bool, str]:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is required"
    
    # Check if it starts with expected prefix
    if not api_key.startswith('dq_'):
        return False, "Invalid API key format"
    
    # Check length (prefix + base64 characters)
    if len(api_key) < 10:
        return False, "API key too short"
    
    return True, ""
