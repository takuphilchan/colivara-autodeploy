"""
Flask Application - Document Q&A Web Interface
Main application with authentication, document management, and query interface

Usage:
    python app.py
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from flask_session import Session

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models.database import db_manager
from services.auth import auth_service
from routes import auth_bp, main_bp


def create_app(config_name=None):
    """
    Application factory function.
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
        
    Returns:
        Configured Flask application instance
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    config_class = get_config(config_name)
    app.config.from_object(config_class)
    config_class.init_app(app)
    
    # Setup logging
    setup_logging(app)
    
    # Initialize extensions
    Session(app)
    
    # Initialize database
    db_manager.init_database()
    
    # Create default admin user
    auth_service.create_default_admin()
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register context processors
    register_context_processors(app)
    
    app.logger.info(f"Application started in {config_name or 'development'} mode")
    
    return app


def setup_logging(app):
    """Configure application logging"""
    if not app.debug:
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # File handler
        file_handler = RotatingFileHandler(
            'logs/app.log',
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('ColiVara application startup')


def register_blueprints(app):
    """Register Flask blueprints"""
    # Authentication routes
    app.register_blueprint(auth_bp)
    
    # Main application routes
    app.register_blueprint(main_bp)
    
    app.logger.info("Blueprints registered successfully")


def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        from flask import jsonify, request
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Not found'}), 404
        return "Page not found", 404
    
    @app.errorhandler(403)
    def forbidden_error(error):
        from flask import jsonify, request
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Forbidden'}), 403
        return "Forbidden", 403
    
    @app.errorhandler(500)
    def internal_error(error):
        from flask import jsonify, request
        app.logger.error(f'Internal error: {error}')
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
        return "Internal server error", 500
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        from flask import jsonify
        return jsonify({
            'success': False,
            'error': 'File too large. Maximum upload size is 50MB.'
        }), 413


def register_context_processors(app):
    """Register template context processors"""
    
    @app.context_processor
    def inject_user():
        from flask import session
        return dict(
            current_user={
                'id': session.get('user_id'),
                'username': session.get('username'),
                'email': session.get('email'),
                'role': session.get('role'),
                'authenticated': session.get('authenticated', False)
            }
        )


# Create application instance for direct execution
app = create_app(os.environ.get('FLASK_ENV', 'development'))


if __name__ == '__main__':
    # Get configuration
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.logger.info(f"Starting Flask application on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
