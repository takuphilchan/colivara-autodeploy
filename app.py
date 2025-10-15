from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import requests
import json
import logging
import time
import markdown
import threading
import uuid
import socket
import os
import sqlite3
import hashlib
import traceback
import secrets
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from functools import wraps

app = Flask(__name__)

# Enhanced configuration
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_FILE = "app_database.db"
STORAGE_FILE = "queries_data.json"

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'csv', 'xls', 'xlsx', 'ppt', 'pptx', 'md', 'html', 'json'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HARDCODED_API_KEY = 'xhFgoUo3UEdhmlIjtq41ds7QJwDM1Yxo'

# In-memory storage (consider moving to database for production)
query_results = {}
query_lock = threading.Lock()
active_threads = {}
thread_lock = threading.Lock()

# ===== DATABASE FUNCTIONS =====

def init_database():
    """Initialize the SQLite database with required tables"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                api_key TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP NULL
            )
        ''')
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Queries table (persistent storage)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                query_text TEXT NOT NULL,
                api_url TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                result TEXT NULL,
                result_html TEXT NULL,
                error TEXT NULL,
                diagram_base64 TEXT NULL,
                mermaid_syntax TEXT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User collections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                collection_name TEXT NOT NULL,
                backend_url TEXT NOT NULL,
                is_default BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, collection_name)
            )
        ''')
        
        # Activity logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        
        # Create default admin user if none exists
        create_default_admin()
        
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

def create_default_admin():
    """Create default admin user if no users exist"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count == 0:
            admin_password = secrets.token_urlsafe(16)
            password_hash = hash_password(admin_password)
            api_key = generate_api_key()
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role, api_key)
                VALUES (?, ?, ?, ?, ?)
            ''', ('admin', 'admin@example.com', password_hash, 'admin', api_key))
            
            conn.commit()
            logger.info(f"Default admin created - Username: admin, Password: {admin_password}")
            logger.info(f"Admin API Key: {api_key}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating default admin: {str(e)}")

def hash_password(password):
    """Hash password using SHA-256 with salt"""
    salt = secrets.token_hex(32)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def verify_password(password, hashed):
    """Verify password against hash"""
    try:
        salt, stored_hash = hashed.split(':')
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return password_hash == stored_hash
    except:
        return False

def generate_api_key():
    """Generate unique API key for user"""
    return f"dq_{secrets.token_urlsafe(32)}"

def log_activity(user_id, action, details=None):
    """Log user activity"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        user_agent = request.headers.get('User-Agent', 'unknown')
        
        cursor.execute('''
            INSERT INTO activity_logs (user_id, action, details, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, action, details, ip_address, user_agent))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error logging activity: {str(e)}")

# ===== AUTHENTICATION DECORATORS =====

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or not session.get('authenticated'):
            if request.is_json:
                return jsonify({'success': False, 'error': 'Authentication required', 'redirect': '/login'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or not session.get('authenticated'):
            if request.is_json:
                return jsonify({'success': False, 'error': 'Authentication required'}), 401
            return redirect(url_for('login'))
        
        # Check if user is admin
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM users WHERE id = ?", (session['user_id'],))
        user = cursor.fetchone()
        conn.close()
        
        if not user or user[0] != 'admin':
            if request.is_json:
                return jsonify({'success': False, 'error': 'Admin access required'}), 403
            return redirect(url_for('index'))
        
        return f(*args, **kwargs)
    return decorated_function

def get_user_api_key(user_id):
    """Return the global hardcoded API key. User-specific keys are not used."""
    # We intentionally ignore the DB-stored api_key and always return a shared key.
    return HARDCODED_API_KEY

# ===== AUTHENTICATION ROUTES =====

@app.route('/login')
def login():
    """Login page"""
    if 'user_id' in session and session.get('authenticated'):
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register')
def register():
    """Registration page"""
    if 'user_id' in session and session.get('authenticated'):
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    """Handle login API request"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')
        remember_me = data.get('remember_me', False)
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password required'})
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Check if account is locked
        cursor.execute('''
            SELECT id, username, email, password_hash, role, api_key, failed_login_attempts, locked_until
            FROM users WHERE (username = ? OR email = ?) AND is_active = 1
        ''', (username, username))
        
        user = cursor.fetchone()
        
        if not user:
            log_activity(None, 'failed_login', f'Unknown username: {username}')
            return jsonify({'success': False, 'error': 'Invalid credentials'})
        
        user_id, db_username, email, password_hash, role, api_key, failed_attempts, locked_until = user
        
        # Check if account is locked
        if locked_until:
            lock_time = datetime.fromisoformat(locked_until)
            if datetime.now() < lock_time:
                return jsonify({'success': False, 'error': 'Account temporarily locked. Please try again later.'})
            else:
                # Unlock account
                cursor.execute('UPDATE users SET locked_until = NULL, failed_login_attempts = 0 WHERE id = ?', (user_id,))
                conn.commit()
        
        if verify_password(password, password_hash):
            # Successful login
            session.permanent = remember_me
            session['user_id'] = user_id
            session['username'] = db_username
            session['email'] = email
            session['role'] = role
            session['authenticated'] = True
            
            # Reset failed attempts
            cursor.execute('''
                UPDATE users SET failed_login_attempts = 0, locked_until = NULL, last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            log_activity(user_id, 'login', 'Successful login')
            
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user': {
                    'id': user_id,
                    'username': db_username,
                    'email': email,
                    'role': role
                }
            })
        else:
            # Failed login
            failed_attempts += 1
            
            # Lock account after 5 failed attempts for 30 minutes
            if failed_attempts >= 5:
                lock_until = datetime.now() + timedelta(minutes=30)
                cursor.execute('''
                    UPDATE users SET failed_login_attempts = ?, locked_until = ?
                    WHERE id = ?
                ''', (failed_attempts, lock_until.isoformat(), user_id))
            else:
                cursor.execute('UPDATE users SET failed_login_attempts = ? WHERE id = ?', (failed_attempts, user_id))
            
            conn.commit()
            conn.close()
            
            log_activity(user_id, 'failed_login', f'Failed password attempt #{failed_attempts}')
            
            return jsonify({'success': False, 'error': 'Invalid credentials'})
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'success': False, 'error': 'Login failed'})

@app.route('/api/register', methods=['POST'])
def api_register():
    """Handle registration API request"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        
        # Validation
        if not username or not email or not password:
            return jsonify({'success': False, 'error': 'All fields are required'})
        
        if password != confirm_password:
            return jsonify({'success': False, 'error': 'Passwords do not match'})
        
        if len(password) < 8:
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters'})
        
        if len(username) < 3:
            return jsonify({'success': False, 'error': 'Username must be at least 3 characters'})
        
        # Check if user already exists
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        existing_user = cursor.fetchone()
        
        if existing_user:
            conn.close()
            return jsonify({'success': False, 'error': 'Username or email already exists'})
        
        # Create new user
        password_hash = hash_password(password)
        api_key = generate_api_key()
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, api_key, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, password_hash, api_key, 'user'))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        log_activity(user_id, 'register', 'New user registered')
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user': {
                'id': user_id,
                'username': username,
                'email': email,
                'role': 'user'
            }
        })
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'success': False, 'error': 'Registration failed'})

@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    """Handle logout"""
    try:
        log_activity(session.get('user_id'), 'logout', 'User logged out')
        session.clear()
        return jsonify({'success': True, 'message': 'Logged out successfully'})
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({'success': False, 'error': 'Logout failed'})

@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    """Get user profile information"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, email, role, api_key, created_at, last_login
            FROM users WHERE id = ?
        ''', (session['user_id'],))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return jsonify({
                'success': True,
                'user': {
                    'username': user[0],
                    'email': user[1],
                    'role': user[2],
                    'api_key': user[3],
                    'created_at': user[4],
                    'last_login': user[5]
                }
            })
        else:
            return jsonify({'success': False, 'error': 'User not found'})
            
    except Exception as e:
        logger.error(f"Profile error: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to get profile'})

@app.route('/api/profile', methods=['PUT'])
@login_required
def update_profile():
    """Update user profile"""
    try:
        data = request.json
        email = data.get('email', '').strip()
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Get current user data
        cursor.execute('SELECT password_hash FROM users WHERE id = ?', (session['user_id'],))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'success': False, 'error': 'User not found'})
        
        # Update email if provided
        updates = []
        params = []
        
        if email:
            # Check if email is already taken
            cursor.execute('SELECT id FROM users WHERE email = ? AND id != ?', (email, session['user_id']))
            if cursor.fetchone():
                conn.close()
                return jsonify({'success': False, 'error': 'Email already taken'})
            
            updates.append('email = ?')
            params.append(email)
            session['email'] = email
        
        # Update password if provided
        if new_password:
            if not current_password:
                conn.close()
                return jsonify({'success': False, 'error': 'Current password required'})
            
            if not verify_password(current_password, user[0]):
                conn.close()
                return jsonify({'success': False, 'error': 'Invalid current password'})
            
            if len(new_password) < 8:
                conn.close()
                return jsonify({'success': False, 'error': 'New password must be at least 8 characters'})
            
            password_hash = hash_password(new_password)
            updates.append('password_hash = ?')
            params.append(password_hash)
        
        if updates:
            params.append(session['user_id'])
            cursor.execute(f'UPDATE users SET {", ".join(updates)} WHERE id = ?', params)
            conn.commit()
        
        conn.close()
        
        log_activity(session['user_id'], 'profile_update', 'Profile updated')
        
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
        
    except Exception as e:
        logger.error(f"Profile update error: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to update profile'})

# ===== ADMIN ROUTES =====

@app.route('/admin')
@admin_required
def admin_panel():
    """Admin panel page"""
    return render_template('admin.html')

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_users():
    """Get all users for admin"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, role, created_at, last_login, is_active, failed_login_attempts
            FROM users ORDER BY created_at DESC
        ''')
        
        users = cursor.fetchall()
        conn.close()
        
        return jsonify({
            'success': True,
            'users': [{
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'role': user[3],
                'created_at': user[4],
                'last_login': user[5],
                'is_active': bool(user[6]),
                'failed_login_attempts': user[7]
            } for user in users]
        })
        
    except Exception as e:
        logger.error(f"Get users error: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to get users'})

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(user_id):
    """Update user by admin"""
    try:
        data = request.json
        role = data.get('role')
        is_active = data.get('is_active')
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if role in ['user', 'admin']:
            updates.append('role = ?')
            params.append(role)
        
        if isinstance(is_active, bool):
            updates.append('is_active = ?')
            params.append(is_active)
        
        if updates:
            params.append(user_id)
            cursor.execute(f'UPDATE users SET {", ".join(updates)} WHERE id = ?', params)
            conn.commit()
        
        conn.close()
        
        log_activity(session['user_id'], 'admin_update_user', f'Updated user {user_id}')
        
        return jsonify({'success': True, 'message': 'User updated successfully'})
        
    except Exception as e:
        logger.error(f"Update user error: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to update user'})

from flask import request, session, jsonify, g
import sqlite3
from datetime import datetime

# Helper to get DB connection (use your existing function if you have one)
def get_db_connection():
    return sqlite3.connect(DATABASE_FILE)

@app.route('/get_current_user', methods=['GET'])
def get_current_user():
    """
    Return minimal current user info for front-end (header).
    Accepts:
      - session cookie (Flask session)
      - Authorization: Bearer <api_key>
      - Cookie or header 'session_token' that exists in user_sessions
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        user_row = None

        # 1) Session-based auth
        user_id = session.get('user_id') if session.get('authenticated') else None
        if user_id:
            cursor.execute('SELECT id, username, email, role FROM users WHERE id = ?', (user_id,))
            user_row = cursor.fetchone()

        # 2) Bearer API key (Authorization header)
        if not user_row:
            auth_header = request.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                api_key = auth_header.split(' ', 1)[1].strip()
                if api_key:
                    cursor.execute('SELECT id, username, email, role FROM users WHERE api_key = ? AND is_active = 1', (api_key,))
                    user_row = cursor.fetchone()

        # 3) Session token lookup: check header first, then cookie
        if not user_row:
            session_token = request.headers.get('X-Session-Token') or request.cookies.get('session_token')
            if session_token:
                # Make sure session is active & not expired
                now_iso = datetime.utcnow().isoformat()
                cursor.execute('''
                    SELECT u.id, u.username, u.email, u.role
                    FROM user_sessions us
                    JOIN users u ON us.user_id = u.id
                    WHERE us.session_token = ? AND us.is_active = 1 AND (us.expires_at > ?)
                ''', (session_token, now_iso))
                user_row = cursor.fetchone()

        conn.close()

        if not user_row:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401

        user_obj = {
            'id': user_row[0],
            'username': user_row[1],
            'email': user_row[2],
            'role': user_row[3]
        }

        return jsonify({'success': True, 'user': user_obj}), 200

    except Exception as e:
        logger.error(f"get_current_user error: {str(e)}")
        return jsonify({'success': False, 'error': 'Server error'}), 500


# ===== MODIFIED EXISTING ROUTES =====

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/get_system_info')
@login_required
def get_system_info():
    """Get system information including IP address"""
    local_ip = get_local_ip()
    return jsonify({
        'local_ip': local_ip,
        'suggested_url': f"http://{local_ip}:5001/query",
        'default_backend_url': f"http://{local_ip}:5001",
        'user': {
            'username': session.get('username'),
            'role': session.get('role')
        }
    })
@app.route('/query_status/<query_id>')
@login_required
def query_status(query_id):
    user_id = session['user_id']
    
    with query_lock:
        if query_id in query_results:
            query_data = query_results[query_id]
            # Check if user owns this query
            if query_data.get('user_id') != user_id:
                return jsonify({
                    'success': False,
                    'error': 'Access denied'
                }), 403
            
            return jsonify({
                'success': True,
                'data': query_data
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Query not found'
            })

@app.route('/get_all_queries')
@login_required
def get_all_queries():
    """Get all queries for current user"""
    user_id = session['user_id']
    
    with query_lock:
        # Filter queries by user
        user_queries = {
            qid: query for qid, query in query_results.items()
            if query.get('user_id') == user_id
        }
        
        return jsonify({
            'success': True,
            'queries': user_queries
        })

# Add these functions to your first Flask app (database version)

@app.route('/delete_query/<query_id>', methods=['DELETE'])
@login_required
def delete_query(query_id):
    try:
        user_id = session['user_id']
        
        # Check memory first and verify ownership
        with query_lock:
            if query_id in query_results:
                if query_results[query_id].get('user_id') != user_id:
                    return jsonify({'success': False, 'error': 'Access denied'}), 403
                del query_results[query_id]
                logger.info(f"Deleted query {query_id} from memory for user {user_id}")
            else:
                logger.warning(f"Query {query_id} not found in memory")
        
        # Delete from database
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM queries WHERE id = ? AND user_id = ?', (query_id, user_id))
        
        if cursor.rowcount > 0:
            conn.commit()
            conn.close()
            log_activity(user_id, 'delete_query', f'Query {query_id} deleted')
            logger.info(f"Query {query_id} deletion saved to database")
            return jsonify({'success': True, 'message': 'Query deleted successfully'})
        else:
            conn.close()
            return jsonify({'success': False, 'error': 'Query not found or access denied'})
            
    except Exception as e:
        logger.error(f"Error deleting query {query_id}: {str(e)}")
        return jsonify({'success': False, 'error': f'Error deleting query: {str(e)}'})

@app.route('/clear_all_queries', methods=['POST'])
@login_required
def clear_all_queries():
    """Clear all queries for current user"""
    try:
        user_id = session['user_id']
        
        # Clear from memory (only user's queries)
        with query_lock:
            user_query_ids = [qid for qid, query in query_results.items() 
                            if query.get('user_id') == user_id]
            for qid in user_query_ids:
                del query_results[qid]
            query_count = len(user_query_ids)
            logger.info(f"Cleared {query_count} queries from memory for user {user_id}")
        
        # Clear from database
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM queries WHERE user_id = ?', (user_id,))
        db_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        log_activity(user_id, 'clear_all_queries', f'Cleared {db_count} queries')
        logger.info(f"All queries cleared from database for user {user_id}")
        
        return jsonify({
            'success': True, 
            'message': f'All {max(query_count, db_count)} queries cleared successfully'
        })
            
    except Exception as e:
        logger.error(f"Error clearing all queries for user {session['user_id']}: {str(e)}")
        return jsonify({'success': False, 'error': f'Error clearing queries: {str(e)}'})

# ==================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ==================================================
@app.route('/preview_document/<document_name>')
@login_required
def preview_document(document_name):
    """Preview document pages"""
    try:
        user_id = session['user_id']
        
        # Get query parameters
        page_number = request.args.get('page_number', type=int)
        max_pages = request.args.get('max_pages', default=1, type=int)
        collection_name = request.args.get('collection_name', 'default_collection')
        backend_url = request.args.get('backend_url', get_default_backend_url())
        
        # Validate max_pages
        if max_pages < 1 or max_pages > 10:
            return jsonify({
                'success': False,
                'error': 'max_pages must be between 1 and 10'
            })
        
        # Use user's API key
        user_api_key = get_user_api_key(user_id)
        if not user_api_key:
            return jsonify({'success': False, 'error': 'User API key not found'})
        
        # Prepare parameters for backend API
        params = {
            'api_key': user_api_key,
            'collection_name': collection_name,
            'max_pages': max_pages
        }
        
        # Add page_number if specified
        if page_number is not None:
            params['page_number'] = page_number
        
        # Call backend API
        response = requests.get(
            f"{backend_url}/preview/{document_name}",
            params=params,
            timeout=30000000
        )
        
        if response.status_code == 200:
            log_activity(user_id, 'preview_document', f'Previewed {document_name}')
            return jsonify({
                'success': True,
                'data': response.json()
            })
        elif response.status_code == 404:
            return jsonify({
                'success': False,
                'error': f'Document "{document_name}" not found'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Preview failed: {response.text}'
            })
            
    except Exception as e:
        logger.error(f"Error previewing document {document_name}: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Preview error: {str(e)}'
        })

@app.route('/search_documents')
@login_required
def search_documents():
    """Search for documents by name"""
    try:
        user_id = session['user_id']
        query = request.args.get('query', '')
        collection_name = request.args.get('collection_name', 'default_collection')
        
        if not query:
            return jsonify({'success': False, 'error': 'Query parameter is required'})
        
        # Use user's API key
        user_api_key = get_user_api_key(user_id)
        if not user_api_key:
            return jsonify({'success': False, 'error': 'User API key not found'})
        
        backend_url = request.args.get('backend_url', get_default_backend_url())
        
        response = requests.get(
            f"{backend_url}/search-documents",
            params={
                'query': query,
                'api_key': user_api_key,
                'collection_name': collection_name
            },
            timeout=30000000
        )
        
        if response.status_code == 200:
            log_activity(user_id, 'search_documents', f'Searched: {query}')
            return jsonify({
                'success': True,
                'data': response.json()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Search failed: {response.text}'
            })
            
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Search error: {str(e)}'
        })

@app.route('/list_documents')
@login_required
def list_documents():
    """List all documents in collection"""
    try:
        user_id = session['user_id']
        collection_name = request.args.get('collection_name', 'default_collection')
        backend_url = request.args.get('backend_url', get_default_backend_url())
        
        # Use user's API key
        user_api_key = get_user_api_key(user_id)
        if not user_api_key:
            return jsonify({'success': False, 'error': 'User API key not found'})
        
        response = requests.get(
            f"{backend_url}/documents",
            params={
                'api_key': user_api_key,
                'collection_name': collection_name
            },
            timeout=30000000
        )
        
        if response.status_code == 200:
            log_activity(user_id, 'list_documents', f'Listed documents in {collection_name}')
            return jsonify({
                'success': True,
                'data': response.json()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to list documents: {response.text}'
            })
            
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'List error: {str(e)}'
        })

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Upload a single document"""
    try:
        user_id = session['user_id']
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed'})
        
        # Get parameters
        collection_name = request.form.get('collection_name', 'default_collection')
        wait_for_processing = request.form.get('wait_for_processing', 'true').lower() == 'true'
        backend_url = request.form.get('backend_url', get_default_backend_url())
        
        # Use user's API key
        user_api_key = get_user_api_key(user_id)
        if not user_api_key:
            return jsonify({'success': False, 'error': 'User API key not found'})
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Upload to backend API
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f, file.content_type)}
                data = {
                    'api_key': user_api_key,
                    'collection_name': collection_name,
                    'wait_for_processing': wait_for_processing
                }
                
                response = requests.post(
                    f"{backend_url}/upload",
                    files=files,
                    data=data,
                    timeout=300
                )
            
            # Clean up temporary file
            os.remove(filepath)
            
            if response.status_code in [200, 201]:
                log_activity(user_id, 'upload_document', f'Uploaded {filename}')
                return jsonify({
                    'success': True,
                    'data': response.json()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Upload failed: {response.text}'
                })
                
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
            
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Upload error: {str(e)}'
        })

@app.route('/upload_bulk', methods=['POST'])
@login_required
def upload_bulk():
    """Upload documents in bulk from a folder"""
    try:
        user_id = session['user_id']
        data = request.json
        
        if not data or 'folder_path' not in data:
            return jsonify({'success': False, 'error': 'Folder path is required'})
        
        folder_path = data['folder_path']
        collection_name = data.get('collection_name', 'default_collection')
        max_file_size_mb = data.get('max_file_size_mb', 50)
        continue_on_error = data.get('continue_on_error', True)
        max_concurrent = data.get('max_concurrent', 3)
        backend_url = data.get('backend_url', get_default_backend_url())
        
        # Validate folder path
        if not os.path.exists(folder_path):
            return jsonify({'success': False, 'error': 'Folder path does not exist'})
        
        # Use user's API key
        user_api_key = get_user_api_key(user_id)
        if not user_api_key:
            return jsonify({'success': False, 'error': 'User API key not found'})
        
        payload = {
            'folder_path': folder_path,
            'api_key': user_api_key,
            'collection_name': collection_name,
            'max_file_size_mb': max_file_size_mb,
            'continue_on_error': continue_on_error,
            'max_concurrent': max_concurrent
        }
        
        response = requests.post(
            f"{backend_url}/upload-bulk",
            data=payload,
            timeout=60000000  # 10 minutes for bulk upload
        )
        
        if response.status_code == 200:
            log_activity(user_id, 'upload_bulk', f'Bulk upload from {folder_path}')
            return jsonify({
                'success': True,
                'data': response.json()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Bulk upload failed: {response.text}'
            })
            
    except Exception as e:
        logger.error(f"Error in bulk upload: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Bulk upload error: {str(e)}'
        })

@app.route('/delete_document/<filename>', methods=['DELETE'])
@login_required
def delete_document(filename):
    """Delete a specific document"""
    try:
        user_id = session['user_id']
        collection_name = request.args.get('collection_name', 'default_collection')
        backend_url = request.args.get('backend_url', get_default_backend_url())
        
        # Use user's API key
        user_api_key = get_user_api_key(user_id)
        if not user_api_key:
            return jsonify({'success': False, 'error': 'User API key not found'})
        
        response = requests.delete(
            f"{backend_url}/delete/{filename}",
            params={
                'api_key': user_api_key,
                'collection_name': collection_name
            },
            timeout=30000000
        )
        
        if response.status_code in [200, 204]:
            log_activity(user_id, 'delete_document', f'Deleted {filename}')
            return jsonify({
                'success': True,
                'message': f'Document {filename} deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Delete failed: {response.text}'
            })
            
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Delete error: {str(e)}'
        })

@app.route('/delete_matching_documents', methods=['DELETE'])
@login_required
def delete_matching_documents():
    """Delete documents that match a keyword"""
    try:
        user_id = session['user_id']
        keyword = request.args.get('keyword', '')
        collection_name = request.args.get('collection_name', 'default_collection')
        backend_url = request.args.get('backend_url', get_default_backend_url())
        
        if not keyword:
            return jsonify({'success': False, 'error': 'Keyword parameter is required'})
        
        # Use user's API key
        user_api_key = get_user_api_key(user_id)
        if not user_api_key:
            return jsonify({'success': False, 'error': 'User API key not found'})
        
        response = requests.delete(
            f"{backend_url}/delete-matching",
            params={
                'keyword': keyword,
                'api_key': user_api_key,
                'collection_name': collection_name
            },
            timeout=6000000
        )
        
        if response.status_code == 200:
            log_activity(user_id, 'delete_matching_documents', f'Deleted matching: {keyword}')
            return jsonify({
                'success': True,
                'data': response.json()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Delete matching failed: {response.text}'
            })
            
    except Exception as e:
        logger.error(f"Error deleting matching documents: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Delete matching error: {str(e)}'
        })

@app.route('/check_upload_status/<document_name>')
@login_required
def check_upload_status(document_name):
    """Check the upload status of a document"""
    try:
        user_id = session['user_id']
        collection_name = request.args.get('collection_name', 'default_collection')
        backend_url = request.args.get('backend_url', get_default_backend_url())
        
        # Use user's API key
        user_api_key = get_user_api_key(user_id)
        if not user_api_key:
            return jsonify({'success': False, 'error': 'User API key not found'})
        
        response = requests.get(
            f"{backend_url}/upload-status/{document_name}",
            params={
                'api_key': user_api_key,
                'collection_name': collection_name
            },
            timeout=30000000
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Status check failed: {response.text}'
            })
            
    except Exception as e:
        logger.error(f"Error checking upload status: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Status check error: {str(e)}'
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/debug_info')
@login_required
def debug_info():
    """Debug endpoint to check current state for user"""
    user_id = session['user_id']
    
    with query_lock:
        with thread_lock:
            # Filter queries by user
            user_queries = {qid: query for qid, query in query_results.items() 
                          if query.get('user_id') == user_id}
            
            return jsonify({
                'user_id': user_id,
                'username': session.get('username'),
                'query_count': len(user_queries),
                'active_threads': len(active_threads),
                'local_ip': get_local_ip(),
                'default_backend_url': get_default_backend_url(),
                'queries': {qid: {
                    'status': q.get('status', 'unknown'), 
                    'query': q.get('query', '')[:50] + '...' if len(q.get('query', '')) > 50 else q.get('query', '')
                } for qid, q in user_queries.items()}
            })

def log_activity(user_id: int, activity_type: str, message: str):
    """Stub for activity logging. Replace with your real implementation."""
    logger.info(f"activity | user_id={user_id} | {activity_type} | {message}")

# -----------------------
# Main route + background worker
# -----------------------

@app.route('/submit_query', methods=['POST'])
@login_required
def submit_query():
    try:
        logger.debug(f"submit_query | Starting query submission | UserID: {session.get('user_id')}")
        data = request.json
        logger.debug(f"submit_query | Request data: {json.dumps(data, indent=2)}")

        # Generate unique query ID
        query_id = str(uuid.uuid4())
        user_id = session['user_id']
        logger.debug(f"submit_query | Generated QueryID: {query_id} | UserID: {user_id}")

        # Validate required fields
        if not data.get('query'):
            logger.error(f"submit_query | Query field missing | QueryID: {query_id}")
            return jsonify({
                'success': False,
                'error': 'Query field is required'
            })

        if not data.get('api_url'):
            logger.error(f"submit_query | API URL field missing | QueryID: {query_id}")
            return jsonify({
                'success': False,
                'error': 'API URL field is required'
            })

        # Use user's API key instead of hardcoded one
        user_api_key = get_user_api_key(user_id)
        if not user_api_key:
            logger.error(f"submit_query | User API key not found | UserID: {user_id}")
            return jsonify({
                'success': False,
                'error': 'User API key not found'
            })
        logger.debug(f"submit_query | Retrieved user API key | UserID: {user_id}")

        # Prepare API payload with user's API key
        payload = {
            "api_key": user_api_key,
            "query": data['query'],
            "include_next_page": data.get('include_next_page', False),
            "max_additional_pages": int(data.get('max_additional_pages', 1)),
            "allow_general_fallback": data.get('allow_general_fallback', False),
            "similarity_threshold": float(data.get('similarity_threshold', 0.25))
        }
        logger.debug(f"submit_query | Prepared payload | QueryID: {query_id}")

        # Store in database
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO queries (id, user_id, query_text, api_url, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (query_id, user_id, data['query'], data['api_url'], 'pending'))

            conn.commit()
            conn.close()
            logger.debug(f"submit_query | Stored query in database | QueryID: {query_id}")
        except Exception as db_error:
            logger.error(f"submit_query | Database error: {str(db_error)} | QueryID: {query_id}")
            return jsonify({
                'success': False,
                'error': f'Database error: {str(db_error)}'
            })

        # Store initial query info in memory (for real-time updates)
        with query_lock:
            query_results[query_id] = {
                'user_id': user_id,  # Add user_id for access control
                'status': 'pending',
                'query': data['query'],
                'api_url': data['api_url'],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'created_timestamp': time.time()
            }
        logger.debug(f"submit_query | Stored query in memory | QueryID: {query_id}")

        # Start background thread for API call
        try:
            thread = threading.Thread(
                target=call_backend_api,
                args=(data['api_url'], payload, query_id, user_id)
            )
            thread.daemon = True
            thread.start()
            logger.debug(f"submit_query | Started background thread | QueryID: {query_id} | Thread: {thread.name}")
        except Exception as thread_error:
            logger.error(f"submit_query | Thread start error: {str(thread_error)} | QueryID: {query_id}")
            # Update status to error since thread failed to start
            with query_lock:
                if query_id in query_results:
                    query_results[query_id].update({
                        'status': 'error',
                        'error': f'Failed to start processing thread: {str(thread_error)}',
                        'updated_at': datetime.now().isoformat()
                    })

            return jsonify({
                'success': False,
                'error': f'Failed to start processing: {str(thread_error)}'
            })

        log_activity(user_id, 'submit_query', f'Query: {data["query"][:100]}...')
        logger.info(f"submit_query | Query submitted successfully | QueryID: {query_id} | UserID: {user_id}")

        return jsonify({
            'success': True,
            'query_id': query_id,
            'message': 'Query submitted successfully'
        })

    except Exception as e:
        logger.error(f"submit_query | Unexpected error: {str(e)} | Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f"Error submitting query: {str(e)}"
        })


def call_backend_api(api_url: str, payload: dict, query_id: str, user_id: int):
    """Call the backend API with the provided payload - Fixed version"""
    try:
        logger.debug(f"call_backend_api | Starting API call | QueryID: {query_id} | API_URL: {api_url}")

        # Register this thread
        with thread_lock:
            active_threads[query_id] = threading.current_thread()
        logger.debug(f"call_backend_api | Registered thread | QueryID: {query_id} | Thread: {threading.current_thread().name}")

        # Check if query was deleted before we start processing
        with query_lock:
            if query_id not in query_results or query_results[query_id].get('user_id') != user_id:
                logger.warning(f"call_backend_api | Query deleted or access denied before processing | QueryID: {query_id}")
                return

            query_results[query_id]['status'] = 'processing'
            query_results[query_id]['updated_at'] = datetime.now().isoformat()
            logger.debug(f"call_backend_api | Updated status to processing | QueryID: {query_id}")

        # Update database
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute('UPDATE queries SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?', 
                          ('processing', query_id, user_id))
            conn.commit()
            conn.close()
            logger.debug(f"call_backend_api | Updated database status to processing | QueryID: {query_id}")
        except Exception as db_error:
            logger.error(f"call_backend_api | Database update error: {str(db_error)} | QueryID: {query_id}")

        # Make API request with detailed logging and FIXED TIMEOUT
        logger.debug(f"call_backend_api | Making POST request to {api_url} | QueryID: {query_id}")
        logger.debug(f"call_backend_api | Request payload keys: {list(payload.keys())} | QueryID: {query_id}")
        
        # Log the actual payload for debugging (be careful with sensitive data)
        logger.debug(f"call_backend_api | Full payload: {json.dumps(payload, indent=2)} | QueryID: {query_id}")

        start_time = time.time()
        
        # FIXED: Use reasonable timeout (5 minutes instead of 115 days)
        response = requests.post(
            api_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Flask-App/1.0"  # Add user agent
            },
            timeout=300,  # 5 minutes - much more reasonable
            allow_redirects=True  # Handle redirects if needed
        )
        
        end_time = time.time()

        # Log more detailed response information
        logger.debug(f"call_backend_api | API response received | Status: {response.status_code} | Duration: {end_time - start_time:.2f}s | QueryID: {query_id}")
        logger.debug(f"call_backend_api | Response headers: {dict(response.headers)} | QueryID: {query_id}")
        
        # Log response content for debugging (first 1000 chars)
        response_text = response.text[:1000] + "..." if len(response.text) > 1000 else response.text
        logger.debug(f"call_backend_api | Response content preview: {response_text} | QueryID: {query_id}")

        # Check again if query was deleted during processing
        with query_lock:
            if query_id not in query_results or query_results[query_id].get('user_id') != user_id:
                logger.warning(f"call_backend_api | Query deleted during processing | QueryID: {query_id}")
                return

        if response.status_code == 200:
            logger.debug(f"call_backend_api | API returned 200 OK | QueryID: {query_id}")
            try:
                api_response = response.json()
            except json.JSONDecodeError as json_error:
                logger.error(f"call_backend_api | JSON decode error: {str(json_error)} | Response: {response.text} | QueryID: {query_id}")
                handle_api_error(query_id, user_id, f"Invalid JSON response from API: {str(json_error)}")
                return

            # Log response structure for debugging
            logger.debug(f"call_backend_api | API response keys: {list(api_response.keys()) if isinstance(api_response, dict) else 'Not a dict'} | QueryID: {query_id}")

            markdown_result = api_response.get("result", "No result found in API response")
            html_result = markdown.markdown(markdown_result, extensions=['tables', 'fenced_code'])

            with query_lock:
                # Final check before updating
                if query_id in query_results and query_results[query_id].get('user_id') == user_id:
                    update_data = {
                        'status': 'completed',
                        'result': markdown_result,
                        'result_html': html_result,
                        'updated_at': datetime.now().isoformat()
                    }

                    # Add diagram data if present
                    if 'diagram_base64' in api_response:
                        update_data['diagram_base64'] = api_response['diagram_base64']
                        logger.debug(f"call_backend_api | Found diagram data | QueryID: {query_id}")

                    # Extract mermaid_syntax from source_info or full_response
                    mermaid_found = False
                    if 'source_info' in api_response and isinstance(api_response['source_info'], dict):
                        if 'mermaid_syntax' in api_response['source_info']:
                            update_data['mermaid_syntax'] = api_response['source_info']['mermaid_syntax']
                            mermaid_found = True
                    elif 'full_response' in api_response and isinstance(api_response['full_response'], dict):
                        if 'mermaid_syntax' in api_response['full_response']:
                            update_data['mermaid_syntax'] = api_response['full_response']['mermaid_syntax']
                            mermaid_found = True

                    if mermaid_found:
                        logger.debug(f"call_backend_api | Found mermaid syntax | QueryID: {query_id}")

                    query_results[query_id].update(update_data)
                    logger.debug(f"call_backend_api | Updated query results in memory | QueryID: {query_id}")

                    # Update database
                    try:
                        conn = sqlite3.connect(DATABASE_FILE)
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE queries SET status = ?, result = ?, result_html = ?, 
                                             diagram_base64 = ?, mermaid_syntax = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ? AND user_id = ?
                        ''', ('completed', markdown_result, html_result, 
                              update_data.get('diagram_base64'), update_data.get('mermaid_syntax'),
                              query_id, user_id))
                        conn.commit()
                        conn.close()
                        logger.debug(f"call_backend_api | Updated database with results | QueryID: {query_id}")
                    except Exception as db_error:
                        logger.error(f"call_backend_api | Database update error: {str(db_error)} | QueryID: {query_id}")

            logger.info(f"call_backend_api | Query completed successfully | QueryID: {query_id}")

        else:
            # ENHANCED: More detailed error logging for non-200 responses
            error_details = f"API returned status {response.status_code}"
            
            # Try to get more details from response
            try:
                if response.headers.get('content-type', '').startswith('application/json'):
                    error_json = response.json()
                    if 'error' in error_json:
                        error_details += f": {error_json['error']}"
                    elif 'message' in error_json:
                        error_details += f": {error_json['message']}"
                    else:
                        error_details += f": {error_json}"
                else:
                    error_details += f": {response.text[:500]}"  # Limit error text length
            except:
                error_details += f": {response.text[:500]}"
                
            logger.error(f"call_backend_api | {error_details} | QueryID: {query_id}")

            with query_lock:
                if query_id in query_results and query_results[query_id].get('user_id') == user_id:
                    query_results[query_id].update({
                        'status': 'error',
                        'error': error_details,
                        'updated_at': datetime.now().isoformat()
                    })

                    # Update database
                    try:
                        conn = sqlite3.connect(DATABASE_FILE)
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE queries SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ? AND user_id = ?
                        ''', ('error', error_details, query_id, user_id))
                        conn.commit()
                        conn.close()
                        logger.debug(f"call_backend_api | Updated database with error | QueryID: {query_id}")
                    except Exception as db_error:
                        logger.error(f"call_backend_api | Database error update failed: {str(db_error)} | QueryID: {query_id}")

    except requests.exceptions.Timeout:
        error_message = "API request timed out after 5 minutes"
        logger.error(f"call_backend_api | {error_message} | QueryID: {query_id}")
        handle_api_error(query_id, user_id, error_message)

    except requests.exceptions.ConnectionError as e:
        error_message = f"Failed to connect to API (connection error): {str(e)}"
        logger.error(f"call_backend_api | {error_message} | QueryID: {query_id}")
        handle_api_error(query_id, user_id, error_message)

    except requests.exceptions.RequestException as e:
        error_message = f"Request failed: {str(e)}"
        logger.error(f"call_backend_api | {error_message} | QueryID: {query_id} | Traceback: {traceback.format_exc()}")
        handle_api_error(query_id, user_id, error_message)

    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logger.error(f"call_backend_api | {error_message} | QueryID: {query_id} | Traceback: {traceback.format_exc()}")
        handle_api_error(query_id, user_id, error_message)

    finally:
        # Unregister thread
        with thread_lock:
            if query_id in active_threads:
                del active_threads[query_id]
                logger.debug(f"call_backend_api | Unregistered thread | QueryID: {query_id}")

def handle_api_error(query_id: str, user_id: int, error_message: str):
    """Helper function to handle API errors consistently"""
    try:
        with query_lock:
            if query_id in query_results and query_results[query_id].get('user_id') == user_id:
                query_results[query_id].update({
                    'status': 'error',
                    'error': error_message,
                    'updated_at': datetime.now().isoformat()
                })

                # Update database
                conn = sqlite3.connect(DATABASE_FILE)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE queries SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND user_id = ?
                ''', ('error', error_message, query_id, user_id))
                conn.commit()
                conn.close()
    except Exception as e:
        logger.error(f"handle_api_error | Failed to update error status: {str(e)} | QueryID: {query_id}")

# Keep existing utility functions
import subprocess
import socket

def get_local_ip():
    """
    Get the local IP address using the same method as the bash script.
    Dynamically adapts to network conditions (online/offline/different networks).
    """
    try:
        # Method 1: Use same approach as bash script (hostname -I | awk '{print $1}')
        result = subprocess.run(
            ['hostname', '-I'], 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=3
        )
        ips = result.stdout.strip().split()
        
        # Return first non-localhost IP (same as awk '{print $1}')
        for ip in ips:
            if not ip.startswith('127.') and '.' in ip:
                # Validate it's a proper IPv4 address
                socket.inet_aton(ip)
                return ip
        
        # If only localhost found, return first IP anyway
        return ips[0] if ips else None
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, OSError, socket.error):
        pass
    
    try:
        # Method 2: Fallback - socket method for outbound connections
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
        
    except Exception:
        pass
    
    try:
        # Method 3: Get from actual network interfaces (no external dependency)
        import os
        result = subprocess.run(
            ['ip', 'addr', 'show'], 
            capture_output=True, 
            text=True, 
            timeout=3
        )
        
        for line in result.stdout.split('\n'):
            if 'inet ' in line and 'scope global' in line:
                ip = line.strip().split()[1].split('/')[0]
                if not ip.startswith('127.'):
                    socket.inet_aton(ip)  # Validate
                    return ip
                    
    except Exception:
        pass
    
    # Ultimate fallback
    return "127.0.0.1"

def get_default_backend_url():
    """Get the default backend URL using local IP"""
    local_ip = get_local_ip()
    return f"http://{local_ip}:8001/v1"

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    # Initialize database on startup
    init_database()
    
    app.run(host='0.0.0.0', port=5000, debug=True)