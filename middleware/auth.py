"""
Authentication and authorization middleware.
Provides decorators for protecting routes and checking permissions.
"""
import logging
from functools import wraps
from flask import session, redirect, url_for, jsonify, request
from models.database import User, db_manager

logger = logging.getLogger(__name__)


def login_required(f):
    """
    Decorator to require login for a route.
    Redirects to login page if not authenticated.
    
    Usage:
        @app.route('/protected')
        @login_required
        def protected_route():
            return "Protected content"
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or not session.get('authenticated'):
            if request.is_json or request.headers.get('Accept') == 'application/json':
                return jsonify({
                    'success': False,
                    'error': 'Authentication required',
                    'redirect': '/login'
                }), 401
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """
    Decorator to require admin role for a route.
    Checks if user is authenticated and has admin role.
    
    Usage:
        @app.route('/admin')
        @admin_required
        def admin_route():
            return "Admin content"
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if authenticated
        if 'user_id' not in session or not session.get('authenticated'):
            if request.is_json or request.headers.get('Accept') == 'application/json':
                return jsonify({
                    'success': False,
                    'error': 'Authentication required'
                }), 401
            return redirect(url_for('auth.login'))
        
        # Check if user is admin
        user_model = User(db_manager)
        user = user_model.get_by_id(session['user_id'])
        
        if not user or user['role'] != 'admin':
            if request.is_json or request.headers.get('Accept') == 'application/json':
                return jsonify({
                    'success': False,
                    'error': 'Admin access required'
                }), 403
            return redirect(url_for('main.index'))
        
        return f(*args, **kwargs)
    return decorated_function


def role_required(*roles):
    """
    Decorator to require specific role(s) for a route.
    
    Args:
        *roles: One or more role names (e.g., 'admin', 'user', 'moderator')
    
    Usage:
        @app.route('/moderator')
        @role_required('admin', 'moderator')
        def moderator_route():
            return "Moderator content"
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if authenticated
            if 'user_id' not in session or not session.get('authenticated'):
                if request.is_json or request.headers.get('Accept') == 'application/json':
                    return jsonify({
                        'success': False,
                        'error': 'Authentication required'
                    }), 401
                return redirect(url_for('auth.login'))
            
            # Check if user has required role
            user_model = User(db_manager)
            user = user_model.get_by_id(session['user_id'])
            
            if not user or user['role'] not in roles:
                if request.is_json or request.headers.get('Accept') == 'application/json':
                    return jsonify({
                        'success': False,
                        'error': f'Access denied. Required role: {", ".join(roles)}'
                    }), 403
                return redirect(url_for('main.index'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def api_key_required(f):
    """
    Decorator to require API key authentication.
    Checks Authorization header for Bearer token.
    
    Usage:
        @app.route('/api/data')
        @api_key_required
        def api_data():
            return jsonify({'data': 'value'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({
                'success': False,
                'error': 'API key required'
            }), 401
        
        # Extract token
        try:
            scheme, token = auth_header.split(' ', 1)
            if scheme.lower() != 'bearer':
                return jsonify({
                    'success': False,
                    'error': 'Invalid authentication scheme. Use Bearer token'
                }), 401
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid Authorization header format'
            }), 401
        
        # Verify API key (simplified - in production, validate against database)
        from config.settings import Config
        if token != Config.HARDCODED_API_KEY:
            return jsonify({
                'success': False,
                'error': 'Invalid API key'
            }), 401
        
        return f(*args, **kwargs)
    return decorated_function


def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    """
    Decorator to rate limit requests.
    Simple in-memory rate limiting (use Redis for production).
    
    Args:
        max_requests: Maximum number of requests allowed
        window_seconds: Time window in seconds
    
    Usage:
        @app.route('/api/search')
        @rate_limit(max_requests=10, window_seconds=60)
        def search():
            return jsonify({'results': []})
    """
    from collections import defaultdict
    from time import time
    
    # In-memory storage (consider using Redis in production)
    request_counts = defaultdict(list)
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get client identifier
            client_id = session.get('user_id', request.remote_addr)
            current_time = time()
            
            # Clean old requests
            request_counts[client_id] = [
                req_time for req_time in request_counts[client_id]
                if current_time - req_time < window_seconds
            ]
            
            # Check rate limit
            if len(request_counts[client_id]) >= max_requests:
                return jsonify({
                    'success': False,
                    'error': 'Rate limit exceeded. Try again later.'
                }), 429
            
            # Add current request
            request_counts[client_id].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_json(*required_fields):
    """
    Decorator to validate JSON request body.
    
    Args:
        *required_fields: Required field names in JSON
    
    Usage:
        @app.route('/api/submit', methods=['POST'])
        @validate_json('name', 'email')
        def submit():
            data = request.json
            return jsonify({'success': True})
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Request must be JSON'
                }), 400
            
            data = request.json
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return jsonify({
                    'success': False,
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator
