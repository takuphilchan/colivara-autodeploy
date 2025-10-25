"""
Configuration settings for the Flask application.
Centralizes all configuration values with environment variable support.
"""
import os
import secrets
from datetime import timedelta
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    """Base configuration class with default settings"""
    
    # Flask Core Settings
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    TESTING = False
    
    # Session Configuration
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=int(os.environ.get('SESSION_LIFETIME_HOURS', 24)))
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Database Configuration
    DATA_DIR = BASE_DIR / 'data'
    DATABASE_FILE = os.environ.get('DATABASE_FILE', str(DATA_DIR / 'app_database.db'))
    STORAGE_FILE = os.environ.get('STORAGE_FILE', str(DATA_DIR / 'queries_data.json'))
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    ALLOWED_EXTENSIONS = {
        'pdf', 'doc', 'docx', 'txt', 'png', 'jpg', 'jpeg', 
        'tiff', 'bmp', 'csv', 'xls', 'xlsx', 'ppt', 'pptx', 
        'md', 'html', 'json'
    }
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_UPLOAD_SIZE', 50)) * 1024 * 1024  # MB to bytes
    
    # API Configuration
    HARDCODED_API_KEY = os.environ.get('API_KEY', 'xhFgoUo3UEdhmlIjtq41ds7QJwDM1Yxo')
    DEFAULT_BACKEND_URL = os.environ.get('BACKEND_URL', 'http://127.0.0.1:8001/v1')
    
    # Security Settings
    MAX_LOGIN_ATTEMPTS = int(os.environ.get('MAX_LOGIN_ATTEMPTS', 5))
    ACCOUNT_LOCK_DURATION_MINUTES = int(os.environ.get('ACCOUNT_LOCK_DURATION', 30))
    MIN_PASSWORD_LENGTH = int(os.environ.get('MIN_PASSWORD_LENGTH', 8))
    MIN_USERNAME_LENGTH = int(os.environ.get('MIN_USERNAME_LENGTH', 3))
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/app.log')
    LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 10485760))  # 10MB
    LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 5))
    
    # Thread Pool Configuration
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 10))
    QUERY_TIMEOUT = int(os.environ.get('QUERY_TIMEOUT', 300))  # seconds
    
    # Rate Limiting (if implemented)
    RATELIMIT_ENABLED = os.environ.get('RATELIMIT_ENABLED', 'False').lower() == 'true'
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '100 per hour')
    
    @staticmethod
    def init_app(app):
        """Initialize application with this config"""
        # Ensure required directories exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(Config.LOG_FILE) if os.path.dirname(Config.LOG_FILE) else 'logs', exist_ok=True)
        os.makedirs('mermaid_diagrams', exist_ok=True)


class DevelopmentConfig(Config):
    """Development-specific configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production-specific configuration"""
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Production-specific initialization
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Setup file logging
        if not app.debug:
            file_handler = RotatingFileHandler(
                cls.LOG_FILE,
                maxBytes=cls.LOG_MAX_BYTES,
                backupCount=cls.LOG_BACKUP_COUNT
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)


class TestingConfig(Config):
    """Testing-specific configuration"""
    TESTING = True
    DATABASE_FILE = ':memory:'  # Use in-memory database for tests
    WTF_CSRF_ENABLED = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
