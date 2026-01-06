-- =============================================
-- AetherMind Autonomous Goal Tracking Schema
-- =============================================
-- This schema enables persistent, autonomous task completion
-- Goals survive server restarts and continue working in background

-- Drop existing tables if recreating
DROP TABLE IF EXISTS goals CASCADE;

-- Goals Table
-- Stores high-level user goals with subtasks and status
CREATE TABLE goals (
    goal_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'retrying', 'blocked')),
    priority INTEGER NOT NULL DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    subtasks JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_goals_user_id ON goals(user_id);
CREATE INDEX idx_goals_status ON goals(status);
CREATE INDEX idx_goals_priority ON goals(priority DESC);
CREATE INDEX idx_goals_created_at ON goals(created_at DESC);
CREATE INDEX idx_goals_user_status ON goals(user_id, status);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at
CREATE TRIGGER update_goals_updated_at BEFORE UPDATE ON goals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (RLS)
ALTER TABLE goals ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only see their own goals
CREATE POLICY goals_user_policy ON goals
    FOR ALL
    USING (user_id = current_setting('app.user_id', TRUE));

-- RLS Policy: System worker can access all goals (for background processing)
CREATE POLICY goals_system_policy ON goals
    FOR ALL
    USING (current_setting('app.is_system_worker', TRUE)::boolean = TRUE);

-- Comment documentation
COMMENT ON TABLE goals IS 'Persistent storage for user goals with autonomous completion tracking';
COMMENT ON COLUMN goals.goal_id IS 'Unique goal identifier (UUID)';
COMMENT ON COLUMN goals.user_id IS 'User who created the goal';
COMMENT ON COLUMN goals.description IS 'High-level goal description';
COMMENT ON COLUMN goals.status IS 'Current execution status: pending, in_progress, completed, failed, retrying, blocked';
COMMENT ON COLUMN goals.priority IS 'Priority level (1-10, higher = more urgent)';
COMMENT ON COLUMN goals.subtasks IS 'Array of subtask objects with execution details';
COMMENT ON COLUMN goals.metadata IS 'Additional context: domain, complexity, etc.';
COMMENT ON COLUMN goals.created_at IS 'Goal creation timestamp';
COMMENT ON COLUMN goals.updated_at IS 'Last update timestamp (auto-updated)';

-- Example subtask structure (stored in goals.subtasks JSONB):
/*
{
    "subtask_id": "uuid",
    "goal_id": "uuid",
    "description": "Install Flask package",
    "action_type": "aether-install",
    "action_params": {
        "content": "",
        "attributes": {"packages": ["flask"]}
    },
    "dependencies": [],
    "status": "pending",
    "attempt_count": 0,
    "max_attempts": 3,
    "execution_result": {
        "success": true,
        "result": "Flask installed successfully",
        "output": "Successfully installed Flask-3.0.0",
        "error": null,
        "metadata": {"execution_time": 2.5}
    },
    "error_message": null,
    "created_at": "2026-01-05T19:00:00Z",
    "updated_at": "2026-01-05T19:00:05Z"
}
*/

-- Sample query to get pending goals ordered by priority
-- SELECT * FROM goals 
-- WHERE status IN ('pending', 'in_progress', 'retrying') 
-- ORDER BY priority DESC, created_at ASC;

-- Sample query to get goal progress
-- SELECT 
--     goal_id,
--     description,
--     status,
--     jsonb_array_length(subtasks) as total_subtasks,
--     (SELECT COUNT(*) FROM jsonb_array_elements(subtasks) sub 
--      WHERE sub->>'status' = 'completed') as completed_subtasks,
--     (SELECT COUNT(*) FROM jsonb_array_elements(subtasks) sub 
--      WHERE sub->>'status' = 'failed') as failed_subtasks
-- FROM goals
-- WHERE user_id = 'user123';
