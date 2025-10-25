"""
Core configuration for FastAPI Document Q&A API.
Centralizes all settings and configuration values.
"""
import os
import socket
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Information
    API_TITLE: str = "Document Q&A API"
    API_DESCRIPTION: str = "API for document management and querying with fixed configurations"
    API_VERSION: str = "2.0.0"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 5001
    DEBUG: bool = False
    
    # Ollama Configuration
    OLLAMA_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "qwen2.5vl:32b"
    
    # MinIO Configuration
    MINIO_ENDPOINT: Optional[str] = None  # Auto-detected if not set
    MINIO_ACCESS_KEY: str = "miniokey"
    MINIO_SECRET_KEY: str = "miniosecret"
    MINIO_BUCKET: str = "colivara"
    MINIO_SECURE: bool = False
    
    # ColiVara Configuration
    COLIVARA_BASE_URL: Optional[str] = None  # Auto-detected if not set
    COLIVARA_BASE_URL_QUERY: Optional[str] = None  # Auto-detected if not set
    DEFAULT_COLLECTION: str = "default_collection"
    
    # Authentication
    API_KEY: str = "xhFgoUo3UEdhmlIjtq41ds7QJwDM1Yxo"
    
    # Query Processing
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.25
    DEFAULT_MAX_ADDITIONAL_PAGES: int = 1
    MAX_DOCUMENTS: int = 3
    PAGES_PER_DOCUMENT: int = 2
    QUERY_TIMEOUT: int = 300  # seconds
    
    # Table Processing
    TABLE_PIPE_RATIO: float = 0.5
    TABLE_COLON_RATIO: float = 0.6
    TABLE_SPACE_RATIO: float = 0.4
    TABLE_CSV_RATIO: float = 0.5
    TABLE_MIN_LINES: int = 2
    TABLE_MERGE_THRESHOLD: float = 0.1
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: set = {
        'pdf', 'doc', 'docx', 'txt', 'png', 'jpg', 'jpeg',
        'tiff', 'bmp', 'csv', 'xls', 'xlsx', 'ppt', 'pptx',
        'md', 'html', 'json'
    }
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Paths
    MERMAID_OUTPUT_DIR: Path = Path("mermaid_diagrams")
    TEMP_DIR: Path = Path("/tmp")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Security
    ALLOWED_HOSTS: list = ["*"]
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables from Flask app
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect local IP and set dependent URLs
        if not self.MINIO_ENDPOINT or not self.COLIVARA_BASE_URL:
            local_ip = self.get_local_ip()
            
            if not self.MINIO_ENDPOINT:
                self.MINIO_ENDPOINT = f"{local_ip}:9000"
            
            if not self.COLIVARA_BASE_URL:
                self.COLIVARA_BASE_URL = f"http://{local_ip}:8001/v1"
            
            if not self.COLIVARA_BASE_URL_QUERY:
                self.COLIVARA_BASE_URL_QUERY = f"http://{local_ip}:8001"
        
        # Ensure directories exist
        self.MERMAID_OUTPUT_DIR.mkdir(exist_ok=True)
    
    @staticmethod
    def get_local_ip() -> str:
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
    
    def get_minio_config(self) -> dict:
        """Get MinIO client configuration"""
        return {
            "endpoint": self.MINIO_ENDPOINT,
            "access_key": self.MINIO_ACCESS_KEY,
            "secret_key": self.MINIO_SECRET_KEY,
            "secure": self.MINIO_SECURE
        }


# Create global settings instance
settings = Settings()
