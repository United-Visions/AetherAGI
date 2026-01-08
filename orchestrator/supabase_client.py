"""
Supabase Client for AetherMind
Handles database operations for API keys, user data, and sessions
"""

import os
from supabase import create_client, Client
from loguru import logger

class SupabaseClient:
    """Singleton Supabase client for database operations"""
    
    _instance = None
    _client: Client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SupabaseClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize Supabase client with credentials from environment"""
        # Try multiple environment variable names for flexibility
        url = os.environ.get("SB_URL") or os.environ.get("SUPABASE_URL")
        key = os.environ.get("SB_SECRET_KEY") or os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
        
        if not url or not key:
            logger.warning("⚠️  Supabase credentials not found. Database features disabled.")
            logger.info("Set SB_URL and SB_SECRET_KEY in environment to enable Supabase")
            self._client = None
            return
        
        try:
            self._client = create_client(url, key)
            logger.info(f"✅ Supabase client initialized: {url}")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    @property
    def client(self) -> Client:
        """Get the Supabase client instance"""
        if self._client is None:
            logger.warning("Supabase client not initialized - database features disabled")
        return self._client
    
    async def create_tables_if_not_exist(self):
        """
        Create necessary tables in Supabase if they don't exist.
        This should be run once during initial setup.
        """
        # SQL to create api_keys table
        create_api_keys_table = """
        CREATE TABLE IF NOT EXISTS api_keys (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id TEXT NOT NULL,
            key_hash TEXT NOT NULL UNIQUE,
            role TEXT NOT NULL DEFAULT 'free',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_used TIMESTAMPTZ,
            metadata JSONB DEFAULT '{}'::jsonb,
            revoked BOOLEAN DEFAULT FALSE,
            CONSTRAINT valid_role CHECK (role IN ('free', 'pro', 'enterprise', 'admin'))
        );
        
        CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
        CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
        CREATE INDEX IF NOT EXISTS idx_api_keys_revoked ON api_keys(revoked) WHERE NOT revoked;
        """
        
        try:
            # Execute raw SQL via Supabase RPC or use client directly
            logger.info("Creating database tables...")
            # Note: Supabase Python client doesn't directly support raw SQL
            # You need to create these tables via Supabase Dashboard SQL Editor
            # or use a PostgreSQL client
            logger.warning("⚠️  Please create tables via Supabase Dashboard SQL Editor:")
            logger.warning(create_api_keys_table)
            return True
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False

# Global instance
supabase = SupabaseClient()
