"""
Setup Supabase Tables for AetherMind
Run this once to create necessary database tables
"""

import os
import sys
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import after loading env vars
from supabase import create_client

def create_tables():
    """Create necessary tables in Supabase"""
    
    url = os.getenv("SB_URL")
    key = os.getenv("SB_SECRET_KEY")
    
    if not url or not key:
        logger.error("SB_URL or SB_SECRET_KEY not found in environment")
        return False
    
    try:
        supabase = create_client(url, key)
        logger.info(f"Connected to Supabase: {url}")
        
        # SQL to create api_keys table with all required columns
        create_api_keys_table = """
        CREATE TABLE IF NOT EXISTS api_keys (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id TEXT NOT NULL,
            github_username TEXT,
            github_url TEXT,
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
        
        # Execute SQL via RPC function
        # Note: This requires a Supabase Edge Function or direct PostgreSQL connection
        logger.info("📝 SQL to execute in Supabase Dashboard:")
        logger.info("="*80)
        logger.info(create_api_keys_table)
        logger.info("="*80)
        logger.info("\n✅ Please run the above SQL in your Supabase Dashboard:")
        logger.info("   1. Go to https://supabase.com/dashboard/project/_/sql")
        logger.info("   2. Paste the SQL above")
        logger.info("   3. Click 'Run'\n")
        
        # Alternative: Use direct PostgreSQL connection
        db_url = os.getenv("SB_POSTGRESQL_URL")
        if db_url:
            logger.info("🔧 Attempting direct PostgreSQL connection...")
            import psycopg2
            
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            cursor.execute(create_api_keys_table)
            conn.commit()
            
            cursor.close()
            conn.close()
            
            logger.success("✅ Tables created successfully via PostgreSQL!")
            return True
        
        return False
        
    except ImportError as e:
        logger.warning(f"psycopg2 not installed: {e}")
        logger.info("Install with: pip install psycopg2-binary")
        return False
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Setting up Supabase tables for AetherMind...")
    create_tables()
