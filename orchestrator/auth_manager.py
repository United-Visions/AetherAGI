"""
Path: orchestrator/auth_manager.py
Role: Persistent Security Layer.
"""

import secrets
import hashlib
import json
import os
from loguru import logger

class AuthManager:
    def __init__(self, key_file="config/keys.json"):
        self.key_file = key_file
        self.key_store = self._load_keys()

    def _load_keys(self):
        """Loads keys from a file on startup."""
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load keys: {e}")
        return {}

    def _save_keys(self):
        """Saves keys to a file."""
        os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
        with open(self.key_file, "w") as f:
            json.dump(self.key_store, f)

    def generate_api_key(self, user_id: str) -> str:
        prefix = "am_live_"
        random_part = secrets.token_urlsafe(32)
        full_key = f"{prefix}{random_part}"
        
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        self.key_store[key_hash] = user_id
        
        self._save_keys() # Save to file immediately
        logger.info(f"Persistent API key generated for: {user_id}")
        return full_key

    def verify_key(self, provided_key: str) -> str:
        if not provided_key:
            return None
            
        key_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        user_id = self.key_store.get(key_hash)
        return user_id