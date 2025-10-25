"""
Database models and utilities.
Handles all database operations with proper connection management and error handling.
"""
import os
import sqlite3
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from config.settings import Config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, db_file: str = None):
        self.db_file = db_file or Config.DATABASE_FILE
        # Ensure the data directory exists
        db_path = Path(self.db_file)
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize the database with all required tables"""
        try:
            with self.get_connection() as conn:
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
                
                # Queries table
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
                
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise


class User:
    """User model with database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create(self, username: str, email: str, password_hash: str, 
               role: str = 'user', api_key: str = None) -> int:
        """Create a new user"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role, api_key)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, role, api_key))
            return cursor.lastrowid
    
    def get_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ? AND is_active = 1', (username,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE email = ? AND is_active = 1', (email,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_by_username_or_email(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Get user by username or email"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM users 
                WHERE (username = ? OR email = ?) AND is_active = 1
            ''', (identifier, identifier))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
    
    def increment_failed_attempts(self, user_id: int) -> int:
        """Increment failed login attempts and return new count"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET failed_login_attempts = failed_login_attempts + 1 
                WHERE id = ?
            ''', (user_id,))
            cursor.execute('SELECT failed_login_attempts FROM users WHERE id = ?', (user_id,))
            return cursor.fetchone()[0]
    
    def reset_failed_attempts(self, user_id: int):
        """Reset failed login attempts"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET failed_login_attempts = 0, locked_until = NULL 
                WHERE id = ?
            ''', (user_id,))
    
    def lock_account(self, user_id: int, locked_until: str):
        """Lock user account until specified time"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET locked_until = ? WHERE id = ?
            ''', (locked_until, user_id))
    
    def update_profile(self, user_id: int, email: str = None, password_hash: str = None):
        """Update user profile"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if email:
                cursor.execute('UPDATE users SET email = ? WHERE id = ?', (email, user_id))
            if password_hash:
                cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (password_hash, user_id))
    
    def exists(self, username: str = None, email: str = None) -> bool:
        """Check if user exists by username or email"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if username and email:
                cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
            elif username:
                cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            elif email:
                cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            else:
                return False
            return cursor.fetchone() is not None
    
    def count(self) -> int:
        """Get total user count"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM users')
            return cursor.fetchone()[0]


class Query:
    """Query model with database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create(self, query_id: str, user_id: int, query_text: str, api_url: str) -> str:
        """Create a new query"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO queries (id, user_id, query_text, api_url, status)
                VALUES (?, ?, ?, ?, 'pending')
            ''', (query_id, user_id, query_text, api_url))
            return query_id
    
    def get_by_id(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get query by ID"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM queries WHERE id = ?', (query_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_status(self, query_id: str, status: str, result: str = None, 
                     result_html: str = None, error: str = None,
                     diagram_base64: str = None, mermaid_syntax: str = None):
        """Update query status and results"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE queries 
                SET status = ?, result = ?, result_html = ?, error = ?, 
                    diagram_base64 = ?, mermaid_syntax = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, result, result_html, error, diagram_base64, mermaid_syntax, query_id))
    
    def get_user_queries(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get queries for a specific user"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM queries WHERE user_id = ? 
                ORDER BY created_at DESC LIMIT ?
            ''', (user_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def delete(self, query_id: str):
        """Delete a query"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM queries WHERE id = ?', (query_id,))


class ActivityLog:
    """Activity log model with database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def log(self, user_id: Optional[int], action: str, details: str = None,
            ip_address: str = None, user_agent: str = None):
        """Log an activity"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO activity_logs (user_id, action, details, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, action, details, ip_address, user_agent))
    
    def get_user_activities(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get activities for a specific user"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM activity_logs WHERE user_id = ? 
                ORDER BY created_at DESC LIMIT ?
            ''', (user_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_activities(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent activities across all users"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT al.*, u.username 
                FROM activity_logs al
                LEFT JOIN users u ON al.user_id = u.id
                ORDER BY al.created_at DESC LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]


class Collection:
    """Collection model with database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create(self, user_id: int, collection_name: str, backend_url: str, 
               is_default: bool = False) -> int:
        """Create a new collection"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # If this is default, unset other defaults
            if is_default:
                cursor.execute('''
                    UPDATE user_collections SET is_default = 0 WHERE user_id = ?
                ''', (user_id,))
            
            cursor.execute('''
                INSERT INTO user_collections (user_id, collection_name, backend_url, is_default)
                VALUES (?, ?, ?, ?)
            ''', (user_id, collection_name, backend_url, is_default))
            return cursor.lastrowid
    
    def get_user_collections(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all collections for a user"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM user_collections WHERE user_id = ? 
                ORDER BY is_default DESC, created_at DESC
            ''', (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_default(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user's default collection"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM user_collections 
                WHERE user_id = ? AND is_default = 1
            ''', (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def set_default(self, collection_id: int, user_id: int):
        """Set a collection as default for user"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Unset all defaults
            cursor.execute('''
                UPDATE user_collections SET is_default = 0 WHERE user_id = ?
            ''', (user_id,))
            # Set new default
            cursor.execute('''
                UPDATE user_collections SET is_default = 1 
                WHERE id = ? AND user_id = ?
            ''', (collection_id, user_id))
    
    def delete(self, collection_id: int, user_id: int):
        """Delete a collection"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM user_collections 
                WHERE id = ? AND user_id = ?
            ''', (collection_id, user_id))


# Initialize database manager instance
db_manager = DatabaseManager()
