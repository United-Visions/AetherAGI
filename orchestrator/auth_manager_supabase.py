"""
Enhanced AuthManager using Supabase PostgreSQL for API key storage
Replaces JSON file storage with cloud database
"""

import secrets
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
from loguru import logger
from collections import defaultdict

from orchestrator.supabase_client import supabase

class UserRole(Enum):
    """User roles for RBAC"""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

class Permission(Enum):
    """Granular permissions"""
    # Basic
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    
    # Advanced
    META_CONTROLLER = "meta_controller"
    SELF_MODIFY = "self_modify"
    TOOL_FORGE = "tool_forge"
    
    # Admin
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT = "view_audit"
    MANAGE_KEYS = "manage_keys"

class AuthManagerSupabase:
    """Enterprise authentication manager with Supabase backend"""
    
    # Role -> Permissions mapping
    ROLE_PERMISSIONS = {
        UserRole.FREE: [Permission.READ, Permission.WRITE],
        UserRole.PRO: [
            Permission.READ, Permission.WRITE, Permission.DELETE,
            Permission.META_CONTROLLER, Permission.SELF_MODIFY, Permission.TOOL_FORGE
        ],
        UserRole.ENTERPRISE: [
            Permission.READ, Permission.WRITE, Permission.DELETE,
            Permission.META_CONTROLLER, Permission.SELF_MODIFY, Permission.TOOL_FORGE,
            Permission.VIEW_AUDIT, Permission.MANAGE_KEYS
        ],
        UserRole.ADMIN: [p for p in Permission]  # All permissions
    }
    
    # Rate limits per role (requests per minute)
    RATE_LIMITS = {
        UserRole.FREE: 100,
        UserRole.PRO: 1000,
        UserRole.ENTERPRISE: 10000,
        UserRole.ADMIN: float('inf')
    }
    
    def __init__(self):
        """Initialize AuthManager with Supabase backend"""
        self.db = supabase.client
        self.rate_limit_tracker = defaultdict(list)  # user_id -> [timestamp, ...]
        logger.info("✅ AuthManager initialized with Supabase backend")
    
    def generate_api_key(self, user_id: str, github_username: str, github_url: str = None,
                         role: UserRole = UserRole.PRO, metadata: Dict = None) -> str:
        """
        Generate persistent API key for a user and store in Supabase
        
        Args:
            user_id: Unique user identifier (GitHub username)
            github_username: GitHub username
            github_url: GitHub profile URL
            role: User role (FREE, PRO, ENTERPRISE, ADMIN)
            metadata: Optional metadata (email, name, etc.)
        
        Returns:
            API key string (format: am_live_XXXXX)
        """
        prefix = "am_live_"
        random_part = secrets.token_urlsafe(32)
        full_key = f"{prefix}{random_part}"
        
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        
        # Store in Supabase
        try:
            data = {
                "user_id": user_id,
                "github_username": github_username,
                "github_url": github_url or f"https://github.com/{github_username}",
                "key_hash": key_hash,
                "role": role.value,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "revoked": False
            }
            
            result = self.db.table("api_keys").insert(data).execute()
            
            if result.data:
                logger.info(f"✅ API key generated and stored for user: {user_id} ({role.value})")
                logger.info(f"   Key prefix: {full_key[:20]}..., hash: {key_hash[:16]}...")
                return full_key
            else:
                logger.error(f"Failed to store API key in Supabase")
                raise Exception("Failed to store API key")
                
        except Exception as e:
            logger.error(f"Error generating API key: {e}")
            raise
    
    def verify_api_key(self, provided_key: str) -> Optional[Dict]:
        """
        Verify API key and return user data from Supabase
        
        Returns:
            {
                "user_id": str,
                "role": str,
                "permissions": List[str]
            } or None
        """
        if not provided_key:
            return None
        
        key_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        
        try:
            # Query Supabase for key
            result = self.db.table("api_keys")\
                .select("*")\
                .eq("key_hash", key_hash)\
                .eq("revoked", False)\
                .execute()
            
            if not result.data or len(result.data) == 0:
                logger.warning(f"Invalid API key attempt - key prefix: {provided_key[:20]}..., hash: {key_hash[:16]}...")
                return None
            
            key_data = result.data[0]
            user_id = key_data["user_id"]
            role = UserRole(key_data["role"])
            
            # Update last_used timestamp
            self.db.table("api_keys")\
                .update({"last_used": datetime.utcnow().isoformat()})\
                .eq("key_hash", key_hash)\
                .execute()
            
            # Check rate limit
            if not self._check_rate_limit(user_id, role):
                logger.warning(f"Rate limit exceeded for user: {user_id}")
                return None
            
            logger.info(f"✅ API key verified for user: {user_id} ({role.value})")
            
            return {
                "user_id": user_id,
                "role": role.value,
                "permissions": [p.value for p in self.ROLE_PERMISSIONS[role]]
            }
            
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return None
            
    def verify_key(self, provided_key: str) -> Optional[str]:
        """Convenience method that returns just the user_id or None"""
        user_data = self.verify_api_key(provided_key)
        return user_data["user_id"] if user_data else None
    
    def revoke_api_key(self, key_hash: str) -> bool:
        """Revoke an API key"""
        try:
            result = self.db.table("api_keys")\
                .update({"revoked": True})\
                .eq("key_hash", key_hash)\
                .execute()
            
            if result.data:
                logger.info(f"✅ API key revoked: {key_hash[:16]}...")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error revoking API key: {e}")
            return False

    def revoke_api_key_by_id(self, key_id: str) -> bool:
        """Revoke an API key by its UUID"""
        try:
            result = self.db.table("api_keys")\
                .update({"revoked": True})\
                .eq("id", key_id)\
                .execute()
            
            if result.data:
                logger.info(f"✅ API key revoked by ID: {key_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error revoking API key by ID: {e}")
            return False
    
    def list_user_keys(self, user_id: str) -> List[Dict]:
        """List all keys for a user"""
        try:
            result = self.db.table("api_keys")\
                .select("id, user_id, role, created_at, last_used, revoked")\
                .eq("user_id", user_id)\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error listing user keys: {e}")
            return []
    
    def _check_rate_limit(self, user_id: str, role: UserRole) -> bool:
        """Check if user is within rate limits"""
        now = datetime.utcnow().timestamp()
        limit = self.RATE_LIMITS[role]
        
        if limit == float('inf'):
            return True
        
        # Clean old entries (older than 1 minute)
        self.rate_limit_tracker[user_id] = [
            ts for ts in self.rate_limit_tracker[user_id]
            if now - ts < 60
        ]
        
        # Check limit
        if len(self.rate_limit_tracker[user_id]) >= limit:
            return False
        
        # Add current request
        self.rate_limit_tracker[user_id].append(now)
        return True

# Global instance
AUTH_SUPABASE = AuthManagerSupabase()
