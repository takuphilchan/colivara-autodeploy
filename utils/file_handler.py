"""
File handling utilities.
Handles file upload validation, storage, and management.
"""
import os
import logging
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from typing import Tuple, Optional
from config.settings import Config

logger = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    """
    Check if the file extension is allowed.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (lowercase) or empty string
    """
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return ''


def save_uploaded_file(file: FileStorage, upload_folder: str = None) -> Tuple[bool, str, Optional[str]]:
    """
    Save uploaded file securely.
    
    Args:
        file: FileStorage object from request
        upload_folder: Directory to save file (default: Config.UPLOAD_FOLDER)
        
    Returns:
        Tuple of (success, message, filepath)
    """
    try:
        if not file or not file.filename:
            return False, 'No file provided', None
        
        if file.filename == '':
            return False, 'Empty filename', None
        
        if not allowed_file(file.filename):
            return False, f'File type not allowed. Allowed types: {", ".join(Config.ALLOWED_EXTENSIONS)}', None
        
        # Create secure filename
        filename = secure_filename(file.filename)
        
        # Use provided folder or default
        folder = upload_folder or Config.UPLOAD_FOLDER
        os.makedirs(folder, exist_ok=True)
        
        # Handle duplicate filenames
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            # Add counter to filename
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filepath):
                filename = f"{name}_{counter}{ext}"
                filepath = os.path.join(folder, filename)
                counter += 1
        
        # Save file
        file.save(filepath)
        
        logger.info(f"File saved: {filepath}")
        return True, f'File uploaded successfully: {filename}', filepath
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return False, 'Failed to save file', None


def delete_file(filepath: str) -> Tuple[bool, str]:
    """
    Delete a file safely.
    
    Args:
        filepath: Path to file to delete
        
    Returns:
        Tuple of (success, message)
    """
    try:
        if not os.path.exists(filepath):
            return False, 'File not found'
        
        os.remove(filepath)
        logger.info(f"File deleted: {filepath}")
        return True, 'File deleted successfully'
        
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return False, 'Failed to delete file'


def get_file_size(filepath: str) -> Optional[int]:
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes or None if error
    """
    try:
        return os.path.getsize(filepath)
    except Exception as e:
        logger.error(f"Error getting file size: {str(e)}")
        return None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "10.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def cleanup_old_files(folder: str, max_age_days: int = 30) -> int:
    """
    Clean up files older than specified days.
    
    Args:
        folder: Directory to clean
        max_age_days: Maximum file age in days
        
    Returns:
        Number of files deleted
    """
    import time
    
    try:
        deleted_count = 0
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                
                if file_age > max_age_seconds:
                    os.remove(filepath)
                    deleted_count += 1
                    logger.info(f"Deleted old file: {filepath}")
        
        logger.info(f"Cleaned up {deleted_count} old files from {folder}")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")
        return 0
