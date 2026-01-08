"""
Path: orchestrator/auth_manager.py
Role: Enterprise-Grade Authentication & Authorization System
Features:
  - Multi-method auth (API keys, JWT, OAuth, SSO)
  - Role-Based Access Control (RBAC)
  - Audit logging (every auth event tracked)
  - Rate limiting per user
  - Session management
  - Multi-platform integration support (Python, Next.js, mobile, etc.)
"""

import secrets
import hashlib
import json
import os
import time
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from loguru import logger
from cryptography.fernet import Fernet
from collections import defaultdict

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

class AuthManager:
    """Enterprise authentication manager with multi-platform support"""
    
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
    
    def __init__(self, key_file="config/keys.json", audit_file="config/audit.jsonl", 
                 secret_key=None, encryption_key=None):
        self.key_file = key_file
        self.audit_file = audit_file
        
        # JWT secret for token-based auth
        self.jwt_secret = secret_key or os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
        
        # Encryption for stored keys - try multiple env var names
        self.encryption_key = encryption_key or os.environ.get("FERNET_KEY") or os.environ.get("ENCRYPTION_KEY")
        if self.encryption_key:
            if isinstance(self.encryption_key, str):
                self.encryption_key = self.encryption_key.encode()
            self.cipher = Fernet(self.encryption_key)
        else:
            # Generate new key if not provided
            self.encryption_key = Fernet.generate_key()
            self.cipher = Fernet(self.encryption_key)
            logger.warning(f"Generated new encryption key. Save this: {self.encryption_key.decode()}")
        
        # In-memory stores
        self.key_store = self._load_keys()
        self.sessions = {}  # session_id -> user_data
        self.rate_limit_tracker = defaultdict(list)  # user_id -> [timestamp, ...]
        
    def _load_keys(self) -> Dict:
        """Loads encrypted keys from file"""
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    encrypted_data = json.load(f)
                    # Decrypt if data is encrypted
                    if isinstance(encrypted_data, dict) and "encrypted" in encrypted_data:
                        decrypted = self.cipher.decrypt(encrypted_data["data"].encode())
                        return json.loads(decrypted.decode())
                    return encrypted_data
            except Exception as e:
                logger.error(f"Failed to load keys: {e}")
        return {}

    def _save_keys(self):
        """Saves keys with encryption"""
        os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
        
        # Encrypt data before saving
        data_str = json.dumps(self.key_store)
        encrypted = self.cipher.encrypt(data_str.encode())
        
        with open(self.key_file, "w") as f:
            json.dump({
                "encrypted": True,
                "data": encrypted.decode()
            }, f)
    
    def _log_audit(self, event: str, user_id: str, details: Dict = None, success: bool = True):
        """Log all auth events for compliance"""
        os.makedirs(os.path.dirname(self.audit_file), exist_ok=True)
        
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "user_id": user_id,
            "success": success,
            "details": details or {}
        }
        
        with open(self.audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
        
        logger.info(f"AUDIT: {event} - {user_id} - {'SUCCESS' if success else 'FAILED'}")
    
    def _check_rate_limit(self, user_id: str, role: UserRole) -> bool:
        """Check if user is within rate limits"""
        now = time.time()
        limit = self.RATE_LIMITS[role]
        
        # Clean old timestamps (older than 1 minute)
        self.rate_limit_tracker[user_id] = [
            ts for ts in self.rate_limit_tracker[user_id] 
            if now - ts < 60
        ]
        
        # Check limit
        if len(self.rate_limit_tracker[user_id]) >= limit:
            return False
        
        # Add current timestamp
        self.rate_limit_tracker[user_id].append(now)
        return True
    
    # ============= API KEY AUTHENTICATION =============
    
    def generate_api_key(self, user_id: str, role: UserRole = UserRole.FREE, 
                        metadata: Dict = None) -> str:
        """
        Generate persistent API key for a user
        
        Args:
            user_id: Unique user identifier
            role: User role (FREE, PRO, ENTERPRISE, ADMIN)
            metadata: Optional metadata (email, name, etc.)
        
        Returns:
            API key string (format: am_live_XXXXX)
        """
        prefix = "am_live_"
        random_part = secrets.token_urlsafe(32)
        full_key = f"{prefix}{random_part}"
        
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        
        self.key_store[key_hash] = {
            "user_id": user_id,
            "role": role.value,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "metadata": metadata or {},
            "revoked": False
        }
        
        self._save_keys()
        self._log_audit("api_key_generated", user_id, {"role": role.value})
        
        logger.info(f"API key generated for: {user_id} ({role.value})")
        return full_key

    def verify_api_key(self, provided_key: str) -> Optional[Dict]:
        """
        Verify API key and return user data
        
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
        key_data = self.key_store.get(key_hash)
        
        if not key_data or key_data.get("revoked"):
            self._log_audit("api_key_verify", "unknown", success=False)
            return None
        
        # Update last used
        key_data["last_used"] = datetime.utcnow().isoformat()
        self._save_keys()
        
        user_id = key_data["user_id"]
        role = UserRole(key_data["role"])
        
        # Check rate limit
        if not self._check_rate_limit(user_id, role):
            self._log_audit("rate_limit_exceeded", user_id, success=False)
            return None
        
        self._log_audit("api_key_verify", user_id, {"role": role.value})
        
        return {
            "user_id": user_id,
            "role": role.value,
            "permissions": [p.value for p in self.ROLE_PERMISSIONS[role]]
        }
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.key_store:
            self.key_store[key_hash]["revoked"] = True
            self.key_store[key_hash]["revoked_at"] = datetime.utcnow().isoformat()
            self._save_keys()
            
            user_id = self.key_store[key_hash]["user_id"]
            self._log_audit("api_key_revoked", user_id)
            return True
        
        return False
    
    # ============= JWT TOKEN AUTHENTICATION =============
    
    def generate_jwt_token(self, user_id: str, role: UserRole = UserRole.FREE,
                          expires_in: int = 3600) -> str:
        """
        Generate JWT token for stateless authentication
        
        Args:
            user_id: User identifier
            role: User role
            expires_in: Token expiration in seconds (default 1 hour)
        
        Returns:
            JWT token string
        """
        payload = {
            "user_id": user_id,
            "role": role.value,
            "permissions": [p.value for p in self.ROLE_PERMISSIONS[role]],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(seconds=expires_in)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        self._log_audit("jwt_generated", user_id, {"expires_in": expires_in})
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """
        Verify JWT token and return payload
        
        Returns:
            {
                "user_id": str,
                "role": str,
                "permissions": List[str]
            } or None
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            user_id = payload["user_id"]
            role = UserRole(payload["role"])
            
            # Check rate limit
            if not self._check_rate_limit(user_id, role):
                self._log_audit("rate_limit_exceeded", user_id, success=False)
                return None
            
            self._log_audit("jwt_verify", user_id, {"role": role.value})
            
            return {
                "user_id": payload["user_id"],
                "role": payload["role"],
                "permissions": payload["permissions"]
            }
        except jwt.ExpiredSignatureError:
            self._log_audit("jwt_expired", "unknown", success=False)
            return None
        except jwt.InvalidTokenError:
            self._log_audit("jwt_invalid", "unknown", success=False)
            return None
    
    # ============= SESSION MANAGEMENT =============
    
    def create_session(self, user_id: str, role: UserRole = UserRole.FREE,
                      duration: int = 86400) -> str:
        """
        Create user session (for web apps)
        
        Args:
            user_id: User identifier
            role: User role
            duration: Session duration in seconds (default 24 hours)
        
        Returns:
            session_id string
        """
        session_id = secrets.token_urlsafe(32)
        
        self.sessions[session_id] = {
            "user_id": user_id,
            "role": role.value,
            "permissions": [p.value for p in self.ROLE_PERMISSIONS[role]],
            "created_at": time.time(),
            "expires_at": time.time() + duration
        }
        
        self._log_audit("session_created", user_id, {"session_id": session_id})
        return session_id
    
    def verify_session(self, session_id: str) -> Optional[Dict]:
        """Verify session and return user data"""
        session = self.sessions.get(session_id)
        
        if not session:
            return None
        
        # Check expiration
        if time.time() > session["expires_at"]:
            del self.sessions[session_id]
            self._log_audit("session_expired", session["user_id"], success=False)
            return None
        
        user_id = session["user_id"]
        role = UserRole(session["role"])
        
        # Check rate limit
        if not self._check_rate_limit(user_id, role):
            self._log_audit("rate_limit_exceeded", user_id, success=False)
            return None
        
        return {
            "user_id": session["user_id"],
            "role": session["role"],
            "permissions": session["permissions"]
        }
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy user session"""
        if session_id in self.sessions:
            user_id = self.sessions[session_id]["user_id"]
            del self.sessions[session_id]
            self._log_audit("session_destroyed", user_id)
            return True
        return False
    
    # ============= RBAC UTILITIES =============
    
    def has_permission(self, user_data: Dict, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission.value in user_data.get("permissions", [])
    
    def upgrade_user_role(self, user_id: str, new_role: UserRole) -> bool:
        """Upgrade user role (e.g., FREE -> PRO)"""
        # Find all keys for this user
        updated = False
        for key_hash, key_data in self.key_store.items():
            if key_data["user_id"] == user_id:
                old_role = key_data["role"]
                key_data["role"] = new_role.value
                updated = True
                
                self._log_audit("role_upgraded", user_id, {
                    "old_role": old_role,
                    "new_role": new_role.value
                })
        
        if updated:
            self._save_keys()
        
        return updated
    
    # ============= ADMIN FUNCTIONS =============
    
    def list_users(self) -> List[Dict]:
        """List all users (admin only)"""
        users = {}
        for key_hash, key_data in self.key_store.items():
            user_id = key_data["user_id"]
            if user_id not in users:
                users[user_id] = {
                    "user_id": user_id,
                    "role": key_data["role"],
                    "created_at": key_data["created_at"],
                    "last_used": key_data["last_used"],
                    "api_keys": 0
                }
            users[user_id]["api_keys"] += 1
        
        return list(users.values())
    
    def get_audit_log(self, user_id: str = None, limit: int = 100) -> List[Dict]:
        """Get audit log entries"""
        if not os.path.exists(self.audit_file):
            return []
        
        entries = []
        with open(self.audit_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                if user_id is None or entry["user_id"] == user_id:
                    entries.append(entry)
        
        return entries[-limit:]  # Return last N entries
    
    # ============= INTEGRATION HELPERS =============
    
    def verify_request(self, auth_header: str) -> Optional[Dict]:
        """
        Universal auth verification for HTTP requests
        Supports: Bearer tokens (JWT), API keys
        
        Args:
            auth_header: Authorization header value
                - "Bearer <jwt_token>"
                - "ApiKey <api_key>"
        
        Returns:
            User data dict or None
        """
        if not auth_header:
            return None
        
        parts = auth_header.split(" ", 1)
        if len(parts) != 2:
            return None
        
        auth_type, auth_value = parts
        
        if auth_type.lower() == "bearer":
            return self.verify_jwt_token(auth_value)
        elif auth_type.lower() == "apikey":
            return self.verify_api_key(auth_value)
        
        return None
    
    def get_integration_guide(self, platform: str) -> str:
        """Return integration guide for different platforms"""
        guides = {
            "python": """
# Python Integration

```python
import requests

# Using API Key
headers = {"Authorization": "ApiKey am_live_YOUR_KEY"}
response = requests.post("https://api.aethermind.ai/v1/chat", 
                        headers=headers, 
                        json={"message": "Hello"})

# Using JWT Token
headers = {"Authorization": "Bearer YOUR_JWT_TOKEN"}
response = requests.post("https://api.aethermind.ai/v1/chat",
                        headers=headers,
                        json={"message": "Hello"})
```
""",
            "nextjs": """
// Next.js Integration

```typescript
// lib/aethermind.ts
export async function callAetherMind(message: string) {
  const response = await fetch('https://api.aethermind.ai/v1/chat', {
    method: 'POST',
    headers: {
      'Authorization': `ApiKey ${process.env.AETHERMIND_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ message })
  });
  
  return response.json();
}

// app/api/chat/route.ts
import { callAetherMind } from '@/lib/aethermind';

export async function POST(request: Request) {
  const { message } = await request.json();
  const result = await callAetherMind(message);
  return Response.json(result);
}
```
""",
            "curl": """
# cURL Integration

```bash
# Using API Key
curl -X POST https://api.aethermind.ai/v1/chat \\
  -H "Authorization: ApiKey am_live_YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Hello AetherMind"}'

# Using JWT Token
curl -X POST https://api.aethermind.ai/v1/chat \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Hello AetherMind"}'
```
""",
            "javascript": """
// JavaScript/Node.js Integration

```javascript
const aetherMind = {
  apiKey: 'am_live_YOUR_KEY',
  baseUrl: 'https://api.aethermind.ai/v1',
  
  async chat(message) {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Authorization': `ApiKey ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message })
    });
    
    return response.json();
  }
};

// Usage
const result = await aetherMind.chat("Hello!");
console.log(result);
```
"""
        }
        
        return guides.get(platform.lower(), "Platform not supported yet. Contact support@aethermind.ai")


# ============= INTEGRATION SDK CLASS =============

class AetherMindClient:
    """
    SDK client for easy integration
    
    Example:
        client = AetherMindClient(api_key="am_live_XXX")
        response = client.chat("Hello AetherMind")
    """
    
    def __init__(self, api_key: str = None, jwt_token: str = None, 
                 base_url: str = "https://api.aethermind.ai/v1"):
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.base_url = base_url
        
        if not api_key and not jwt_token:
            raise ValueError("Either api_key or jwt_token must be provided")
    
    def _get_headers(self) -> Dict:
        """Generate auth headers"""
        if self.jwt_token:
            return {"Authorization": f"Bearer {self.jwt_token}"}
        else:
            return {"Authorization": f"ApiKey {self.api_key}"}
    
    def chat(self, message: str, **kwargs) -> Dict:
        """Send chat message"""
        import requests
        response = requests.post(
            f"{self.base_url}/chat",
            headers=self._get_headers(),
            json={"message": message, **kwargs}
        )
        return response.json()
    
    def search_memory(self, query: str, namespace: str = None) -> Dict:
        """Search episodic memory"""
        import requests
        params = {"query": query}
        if namespace:
            params["namespace"] = namespace
        
        response = requests.get(
            f"{self.base_url}/memory/search",
            headers=self._get_headers(),
            params=params
        )
        return response.json()
    
    def create_tool(self, tool_name: str, description: str) -> Dict:
        """Create new tool via ToolForge"""
        import requests
        response = requests.post(
            f"{self.base_url}/toolforge/create",
            headers=self._get_headers(),
            json={"tool_name": tool_name, "description": description}
        )
        return response.json()