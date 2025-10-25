"""
Authentication routes blueprint.
Handles login, registration, logout, and profile management.
"""
from flask import Blueprint, render_template, request, jsonify, session
import logging
from services.auth import auth_service
from middleware.auth import login_required
from utils.validators import (
    validate_email, validate_username, 
    validate_password, validate_password_match,
    sanitize_input
)

logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/login', methods=['GET'])
def login():
    """Login page"""
    if 'user_id' in session and session.get('authenticated'):
        from flask import redirect, url_for
        return redirect(url_for('main.index'))
    return render_template('login.html')


@auth_bp.route('/register', methods=['GET'])
def register():
    """Registration page"""
    if 'user_id' in session and session.get('authenticated'):
        from flask import redirect, url_for
        return redirect(url_for('main.index'))
    return render_template('register.html')


@auth_bp.route('/api/login', methods=['POST'])
def api_login():
    """Handle login API request"""
    try:
        data = request.json
        identifier = sanitize_input(data.get('username', '').strip())
        password = data.get('password', '')
        remember_me = data.get('remember_me', False)
        
        if not identifier or not password:
            return jsonify({'success': False, 'error': 'Username and password required'})
        
        # Authenticate
        success, message, user_data = auth_service.authenticate(
            identifier, password, remember_me
        )
        
        if success:
            # Set session
            session.permanent = remember_me
            session['user_id'] = user_data['id']
            session['username'] = user_data['username']
            session['email'] = user_data['email']
            session['role'] = user_data['role']
            session['authenticated'] = True
            
            return jsonify({
                'success': True,
                'message': message,
                'user': user_data
            })
        else:
            return jsonify({'success': False, 'error': message})
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'success': False, 'error': 'Login failed'})


@auth_bp.route('/api/register', methods=['POST'])
def api_register():
    """Handle registration API request"""
    try:
        data = request.json
        username = sanitize_input(data.get('username', '').strip())
        email = sanitize_input(data.get('email', '').strip())
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        
        # Validate input
        valid, error = validate_username(username)
        if not valid:
            return jsonify({'success': False, 'error': error})
        
        valid, error = validate_email(email)
        if not valid:
            return jsonify({'success': False, 'error': error})
        
        valid, error = validate_password(password)
        if not valid:
            return jsonify({'success': False, 'error': error})
        
        valid, error = validate_password_match(password, confirm_password)
        if not valid:
            return jsonify({'success': False, 'error': error})
        
        # Create user
        success, message, user_id = auth_service.create_user(username, email, password)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'user': {
                    'id': user_id,
                    'username': username,
                    'email': email,
                    'role': 'user'
                }
            })
        else:
            return jsonify({'success': False, 'error': message})
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'success': False, 'error': 'Registration failed'})


@auth_bp.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    """Handle logout"""
    try:
        auth_service.log_activity(session.get('user_id'), 'logout', 'User logged out')
        session.clear()
        return jsonify({'success': True, 'message': 'Logged out successfully'})
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({'success': False, 'error': 'Logout failed'})


@auth_bp.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    """Get user profile information"""
    try:
        from models.database import User, db_manager
        user_model = User(db_manager)
        
        user = user_model.get_by_id(session['user_id'])
        
        if user:
            return jsonify({
                'success': True,
                'user': {
                    'username': user['username'],
                    'email': user['email'],
                    'role': user['role'],
                    'api_key': user['api_key'],
                    'created_at': user['created_at'],
                    'last_login': user['last_login']
                }
            })
        else:
            return jsonify({'success': False, 'error': 'User not found'})
            
    except Exception as e:
        logger.error(f"Profile error: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to get profile'})


@auth_bp.route('/api/profile', methods=['PUT'])
@login_required
def update_profile():
    """Update user profile"""
    try:
        data = request.json
        email = sanitize_input(data.get('email', '').strip())
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        messages = []
        
        # Update email if provided
        if email:
            valid, error = validate_email(email)
            if not valid:
                return jsonify({'success': False, 'error': error})
            
            success, message = auth_service.update_email(session['user_id'], email)
            if success:
                session['email'] = email
                messages.append(message)
            else:
                return jsonify({'success': False, 'error': message})
        
        # Update password if provided
        if new_password:
            if not current_password:
                return jsonify({'success': False, 'error': 'Current password required'})
            
            valid, error = validate_password(new_password)
            if not valid:
                return jsonify({'success': False, 'error': error})
            
            success, message = auth_service.update_password(
                session['user_id'], current_password, new_password
            )
            if success:
                messages.append(message)
            else:
                return jsonify({'success': False, 'error': message})
        
        auth_service.log_activity(session['user_id'], 'profile_update', 'Profile updated')
        
        return jsonify({
            'success': True,
            'message': '; '.join(messages) if messages else 'Profile updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Profile update error: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to update profile'})


@auth_bp.route('/api/current_user', methods=['GET'])
def get_current_user():
    """Get current authenticated user information"""
    try:
        from models.database import User, db_manager
        from datetime import datetime
        
        user_model = User(db_manager)
        user_row = None
        
        # 1) Session-based auth
        user_id = session.get('user_id') if session.get('authenticated') else None
        if user_id:
            user_row = user_model.get_by_id(user_id)
        
        # 2) Bearer API key (Authorization header)
        if not user_row:
            auth_header = request.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                api_key = auth_header.split(' ', 1)[1].strip()
                if api_key:
                    # Simplified: use hardcoded API key
                    from config.settings import Config
                    if api_key == Config.HARDCODED_API_KEY:
                        # Return default admin user if exists
                        user_row = user_model.get_by_username('admin')
        
        if not user_row:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        user_obj = {
            'id': user_row['id'],
            'username': user_row['username'],
            'email': user_row['email'],
            'role': user_row['role']
        }
        
        return jsonify({'success': True, 'user': user_obj}), 200
        
    except Exception as e:
        logger.error(f"get_current_user error: {str(e)}")
        return jsonify({'success': False, 'error': 'Server error'}), 500
