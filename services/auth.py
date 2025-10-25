"""
Authentication service module.
Handles user authentication, password hashing, API key generation, and session management.
"""
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from flask import request
from config.settings import Config
from models.database import User, ActivityLog, db_manager

logger = logging.getLogger(__name__)


class AuthService:
    """Service class for authentication operations"""
    
    def __init__(self):
        self.user_model = User(db_manager)
        self.activity_log = ActivityLog(db_manager)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash password using SHA-256 with salt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password in format 'salt:hash'
        """
        salt = secrets.token_hex(32)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password to verify
            hashed: Hashed password in format 'salt:hash'
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            salt, stored_hash = hashed.split(':')
            password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return password_hash == stored_hash
        except Exception as e:
            logger.error(f"Password verification error: {str(e)}")
            return False
    
    @staticmethod
    def generate_api_key() -> str:
        """
        Generate unique API key for user.
        
        Returns:
            API key string with 'dq_' prefix
        """
        return f"dq_{secrets.token_urlsafe(32)}"
    
    def create_user(self, username: str, email: str, password: str, 
                   role: str = 'user') -> Tuple[bool, str, Optional[int]]:
        """
        Create a new user account.
        
        Args:
            username: Unique username
            email: Unique email address
            password: Plain text password
            role: User role (default: 'user')
            
        Returns:
            Tuple of (success, message, user_id)
        """
        try:
            # Validation
            if len(username) < Config.MIN_USERNAME_LENGTH:
                return False, f'Username must be at least {Config.MIN_USERNAME_LENGTH} characters', None
            
            if len(password) < Config.MIN_PASSWORD_LENGTH:
                return False, f'Password must be at least {Config.MIN_PASSWORD_LENGTH} characters', None
            
            # Check if user exists
            if self.user_model.exists(username=username, email=email):
                return False, 'Username or email already exists', None
            
            # Create user
            password_hash = self.hash_password(password)
            api_key = self.generate_api_key()
            user_id = self.user_model.create(username, email, password_hash, role, api_key)
            
            # Log activity
            self.log_activity(user_id, 'register', 'New user registered')
            
            logger.info(f"User created: {username} (ID: {user_id})")
            return True, 'User created successfully', user_id
            
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return False, 'Failed to create user', None
    
    def authenticate(self, identifier: str, password: str, 
                    remember_me: bool = False) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Authenticate user with username/email and password.
        
        Args:
            identifier: Username or email
            password: Plain text password
            remember_me: Whether to create persistent session
            
        Returns:
            Tuple of (success, message, user_data)
        """
        try:
            # Get user
            user = self.user_model.get_by_username_or_email(identifier)
            
            if not user:
                self.log_activity(None, 'failed_login', f'Unknown identifier: {identifier}')
                return False, 'Invalid credentials', None
            
            # Check if account is locked
            if user['locked_until']:
                lock_time = datetime.fromisoformat(user['locked_until'])
                if datetime.now() < lock_time:
                    return False, 'Account temporarily locked. Please try again later.', None
                else:
                    # Unlock account
                    self.user_model.reset_failed_attempts(user['id'])
            
            # Verify password
            if self.verify_password(password, user['password_hash']):
                # Successful login
                self.user_model.reset_failed_attempts(user['id'])
                self.user_model.update_last_login(user['id'])
                self.log_activity(user['id'], 'login', 'Successful login')
                
                logger.info(f"User logged in: {user['username']} (ID: {user['id']})")
                
                return True, 'Login successful', {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email'],
                    'role': user['role']
                }
            else:
                # Failed login
                failed_attempts = self.user_model.increment_failed_attempts(user['id'])
                
                # Lock account after max attempts
                if failed_attempts >= Config.MAX_LOGIN_ATTEMPTS:
                    lock_until = datetime.now() + timedelta(minutes=Config.ACCOUNT_LOCK_DURATION_MINUTES)
                    self.user_model.lock_account(user['id'], lock_until.isoformat())
                    self.log_activity(user['id'], 'account_locked', 
                                    f'Account locked after {failed_attempts} failed attempts')
                    return False, f'Account locked for {Config.ACCOUNT_LOCK_DURATION_MINUTES} minutes due to multiple failed login attempts', None
                
                self.log_activity(user['id'], 'failed_login', 
                                f'Failed password attempt #{failed_attempts}')
                return False, 'Invalid credentials', None
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False, 'Authentication failed', None
    
    def update_password(self, user_id: int, current_password: str, 
                       new_password: str) -> Tuple[bool, str]:
        """
        Update user password.
        
        Args:
            user_id: User ID
            current_password: Current password for verification
            new_password: New password to set
            
        Returns:
            Tuple of (success, message)
        """
        try:
            user = self.user_model.get_by_id(user_id)
            
            if not user:
                return False, 'User not found'
            
            # Verify current password
            if not self.verify_password(current_password, user['password_hash']):
                return False, 'Current password is incorrect'
            
            # Validate new password
            if len(new_password) < Config.MIN_PASSWORD_LENGTH:
                return False, f'Password must be at least {Config.MIN_PASSWORD_LENGTH} characters'
            
            # Update password
            new_password_hash = self.hash_password(new_password)
            self.user_model.update_profile(user_id, password_hash=new_password_hash)
            
            self.log_activity(user_id, 'password_changed', 'Password updated successfully')
            logger.info(f"Password updated for user ID: {user_id}")
            
            return True, 'Password updated successfully'
            
        except Exception as e:
            logger.error(f"Error updating password: {str(e)}")
            return False, 'Failed to update password'
    
    def update_email(self, user_id: int, new_email: str) -> Tuple[bool, str]:
        """
        Update user email.
        
        Args:
            user_id: User ID
            new_email: New email address
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if email already exists
            existing_user = self.user_model.get_by_email(new_email)
            if existing_user and existing_user['id'] != user_id:
                return False, 'Email already in use'
            
            # Update email
            self.user_model.update_profile(user_id, email=new_email)
            
            self.log_activity(user_id, 'email_changed', f'Email updated to {new_email}')
            logger.info(f"Email updated for user ID: {user_id}")
            
            return True, 'Email updated successfully'
            
        except Exception as e:
            logger.error(f"Error updating email: {str(e)}")
            return False, 'Failed to update email'
    
    def get_user_api_key(self, user_id: int) -> Optional[str]:
        """
        Get user's API key.
        Note: Currently returns hardcoded API key for backward compatibility.
        
        Args:
            user_id: User ID
            
        Returns:
            API key string
        """
        # Return hardcoded API key for now
        # TODO: Implement per-user API keys if needed
        return Config.HARDCODED_API_KEY
    
    def create_default_admin(self) -> Optional[str]:
        """
        Create default admin user if no users exist.
        
        Returns:
            Admin password if created, None otherwise
        """
        try:
            if self.user_model.count() == 0:
                admin_password = secrets.token_urlsafe(16)
                password_hash = self.hash_password(admin_password)
                api_key = self.generate_api_key()
                
                user_id = self.user_model.create(
                    username='admin',
                    email='admin@example.com',
                    password_hash=password_hash,
                    role='admin',
                    api_key=api_key
                )
                
                logger.info(f"Default admin created - Username: admin, Password: {admin_password}")
                logger.info(f"Admin API Key: {api_key}")
                
                return admin_password
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating default admin: {str(e)}")
            return None
    
    def log_activity(self, user_id: Optional[int], action: str, details: str = None):
        """
        Log user activity.
        
        Args:
            user_id: User ID (can be None for unauthenticated actions)
            action: Action type
            details: Additional details
        """
        try:
            ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', 
                                            request.environ.get('REMOTE_ADDR', 'unknown'))
            user_agent = request.headers.get('User-Agent', 'unknown')
            
            self.activity_log.log(user_id, action, details, ip_address, user_agent)
            
        except Exception as e:
            logger.error(f"Error logging activity: {str(e)}")


# Create singleton instance
auth_service = AuthService()
