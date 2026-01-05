-- AetherMind Supabase Database Schema Setup
-- Run this in Supabase SQL Editor to create the tables

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

-- Policy: Users can only read their own keys
CREATE POLICY "Users can view own keys" ON api_keys
    FOR SELECT
    USING (auth.uid()::text = user_id OR auth.role() = 'service_role');

-- Policy: Service role can insert keys
CREATE POLICY "Service can insert keys" ON api_keys
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

-- Policy: Service role can update keys
CREATE POLICY "Service can update keys" ON api_keys
    FOR UPDATE
    USING (auth.role() = 'service_role');

-- Grant permissions to authenticated users
GRANT SELECT ON api_keys TO authenticated;
GRANT ALL ON api_keys TO service_role;

-- Verify table creation
SELECT 
    table_name,
    column_name,
    data_type 
FROM information_schema.columns 
WHERE table_name = 'api_keys'
ORDER BY ordinal_position;
