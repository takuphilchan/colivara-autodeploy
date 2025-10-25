"""Routes package - Flask blueprints for modular routing"""
from .auth import auth_bp
from .main import main_bp

__all__ = ['auth_bp', 'main_bp']
