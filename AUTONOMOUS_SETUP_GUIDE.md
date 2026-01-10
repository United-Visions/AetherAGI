# Quick Setup Guide: AGI Autonomous Completion

## Prerequisites
- Supabase account with database access
- Redis instance (for background worker)
- Google Gemini API key
- AetherMind backend running

## Setup Steps

### 1. Create Supabase Schema
```bash
# Navigate to project root
cd /Users/deion/Desktop/aethermind_universal

# Run schema migration
psql $SUPABASE_URL < scripts/supabase_goals_schema.sql

# Or use Supabase SQL Editor:
# 1. Go to https://supabase.com/dashboard
# 2. Select your project
# 3. Click "SQL Editor"
# 4. Paste contents of scripts/supabase_goals_schema.sql
# 5. Click "Run"
```

### 2. Verify Environment Variables
```bash
# Check all required keys are set
echo "GOOGLE_API_KEY: $GOOGLE_API_KEY"
echo "PINECONE_API_KEY: $PINECONE_API_KEY"
echo "SUPABASE_URL: $SUPABASE_URL"
echo "SB_ANON_KEY: $SB_ANON_KEY"
echo "REDIS_URL: $REDIS_URL"
```

### 3. Install Dependencies (if any missing)
```bash
source .venv/bin/activate
pip install asyncio
```

### 4. Start Backend with Background Worker
```bash
# The background worker starts automatically on server startup
source .venv/bin/activate
python -m uvicorn orchestrator.main_api:app --host 0.0.0.0 --port 8000 --reload
```

Check logs for:
```
🚀 Background worker started for autonomous task completion
```

### 5. Test Autonomous Goal Creation

#### Via API:
```bash
# Get API key first
curl -X POST "http://localhost:8000/admin/generate_key?user_id=test_user&admin_secret=$ADMIN_SECRET"

# Create a goal
curl -X POST http://localhost:8000/v1/goals/create \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Create a simple Python calculator script that can add, subtract, multiply, and divide",
    "priority": 5,
    "metadata": {"domain": "code", "language": "python"}
  }'

# Response will include goal_id
# {"goal_id": "uuid-here", "status": "pending", ...}

# Check status
curl http://localhost:8000/v1/goals/{goal_id}/status \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### Via Frontend:
```javascript
// In chat interface, type:
"@aether-goal Create a Flask todo app with SQLite database"

// System will:
// 1. Parse request
// 2. Create autonomous goal
// 3. Show goal_id and status URL
// 4. Work in background
// 5. Update UI when complete
```

### 6. Monitor Execution

#### Watch Logs:
```bash
# Backend logs
tail -f logs/aethermind.log | grep -E "(GoalTracker|AutonomousAgent|BackgroundWorker)"

# Look for:
# - "📋 Found X pending goals"
# - "🧩 Decomposed goal into X subtasks"
# - "🔧 Executing subtask: ..."
# - "✅ Subtask completed: ..."
# - "✅ Goal completed: ..."
```

#### Check Database:
```sql
-- In Supabase SQL Editor
SELECT 
    goal_id,
    description,
    status,
    jsonb_array_length(subtasks) as total_subtasks,
    (SELECT COUNT(*) FROM jsonb_array_elements(subtasks) sub 
     WHERE sub->>'status' = 'completed') as completed,
    created_at,
    updated_at
FROM goals
ORDER BY created_at DESC;
```

#### Poll API:
```bash
# Watch progress
watch -n 5 "curl -s -H 'Authorization: Bearer YOUR_API_KEY' \
  http://localhost:8000/v1/goals/{goal_id}/status | jq '.progress'"
```

### 7. Verify Self-Healing

Create a goal that will initially fail:

```bash
curl -X POST http://localhost:8000/v1/goals/create \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Create a Flask app that uses a non-existent package called fake_module",
    "priority": 5
  }'
```

Watch logs - you should see:
1. Initial attempt fails
2. Brain diagnoses error: "ModuleNotFoundError: No module named fake_module"
3. Brain generates fix: "Remove import of fake_module"
4. Retry succeeds

## Troubleshooting

### "Background worker not starting"
```bash
# Check if Redis is running
redis-cli ping
# Should respond: PONG

# If not, start Redis:
redis-server

# Or use Docker:
docker run -d -p 6379:6379 redis
```

### "Goals table does not exist"
```bash
# Re-run schema migration
psql $SUPABASE_URL < scripts/supabase_goals_schema.sql

# Or manually in Supabase SQL Editor
```

### "Brain not responding"
```bash
# Test Gemini API directly
python -c "
import os
import litellm
response = litellm.completion(
    model='gemini/gemini-2.5-pro',
    messages=[{'role': 'user', 'content': 'Hello'}],
    api_key=os.getenv('GOOGLE_API_KEY')
)
print(response.choices[0].message.content)
"
```

### "Import errors"
```bash
# Reinstall dependencies
source .venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

1. **Test with simple goal**: "Create a Python hello world script"
2. **Test with complex goal**: "Create a Flask API with 3 routes"
3. **Test error recovery**: Create goal that will fail (missing package, syntax error)
4. **Test persistence**: Create goal, restart server, verify it resumes
5. **Monitor performance**: Check how long goals take to complete

## Integration with Existing Chat

To make goals work from chat interface:

1. Add goal creation button to frontend
2. Parse special syntax like `@aether-goal <description>`
3. Call `/v1/goals/create` endpoint
4. Display goal status in activity feed
5. Stream progress updates via WebSocket

See [docs/AGI_AUTONOMOUS_COMPLETION.md](AGI_AUTONOMOUS_COMPLETION.md) for full documentation.
