"""
Setup script to initialize Supabase database for AetherMind
Run this once to create the api_keys table
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.supabase_client import supabase
from loguru import logger

def setup_database():
    """Create api_keys table in Supabase"""
    
    sql = """
    -- API Keys Table
    CREATE TABLE IF NOT EXISTS api_keys (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id TEXT NOT NULL,
        github_username TEXT NOT NULL,
        github_url TEXT,
        key_hash TEXT NOT NULL UNIQUE,
        role TEXT NOT NULL DEFAULT 'pro',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        last_used TIMESTAMPTZ,
        metadata JSONB DEFAULT '{}'::jsonb,
        revoked BOOLEAN DEFAULT FALSE,
        CONSTRAINT valid_role CHECK (role IN ('free', 'pro', 'enterprise', 'admin'))
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
    CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
    CREATE INDEX IF NOT EXISTS idx_api_keys_github_username ON api_keys(github_username);
    CREATE INDEX IF NOT EXISTS idx_api_keys_revoked ON api_keys(revoked) WHERE NOT revoked;

    -- Enable Row Level Security (RLS)
    ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
    """
    
    print("=" * 60)
    print("SUPABASE DATABASE SETUP")
    print("=" * 60)
    print("\n📋 Please copy and run the following SQL in Supabase SQL Editor:")
    print(f"\n{sql}\n")
    print("=" * 60)
    print("\n🔗 Go to: https://supabase.com/dashboard/project/_/sql/new")
    print("\nAfter running the SQL, press Enter to verify the setup...")
    input()
    
    # Verify table exists
    try:
        result = supabase.client.table("api_keys").select("*").limit(1).execute()
        print("\n✅ SUCCESS! Table 'api_keys' is accessible")
        print(f"   - Connection: {os.getenv('SB_URL')}")
        return True
    except Exception as e:
        print(f"\n❌ ERROR: Could not access table 'api_keys'")
        print(f"   - Error: {e}")
        print("\n💡 Make sure you ran the SQL in Supabase Dashboard first!")
        return False

if __name__ == "__main__":
    setup_database()
