# AetherMind AGI: Autonomous Task Completion System

## Overview

AetherMind now features **full AGI-level autonomous task completion** with self-healing, persistent goal tracking, and background execution. The system can work independently to complete complex tasks, even when users close their browser or the server restarts.

## Key Features

### 1. **Self-Healing Error Recovery**
- Automatically detects action execution failures
- Analyzes error messages to diagnose root causes
- Generates fix strategies using Brain's reasoning
- Retries failed operations with learned corrections
- Never halluc inates success - only bases decisions on actual execution results

### 2. **Persistent Goal Tracking**
- Goals stored in Supabase, survive server restarts
- Automatic decomposition of complex tasks into subtasks
- Dependency management (subtasks execute when dependencies complete)
- Progress tracking with real-time status updates

### 3. **Background Execution**
- Works independently of user sessions
- Continues processing even when browser is closed
- Background worker polls for pending goals every 30 seconds
- Spawns autonomous agents to work on goals concurrently

### 4. **Multi-Turn Execution**
- Complex tasks may take multiple reasoning cycles
- Execution results feed back to Brain in next turn
- Brain sees actual outputs/errors and adjusts approach
- Continuous loop until goal completion

### 5. **Execution Result Feedback**
- Every action tag execution returns detailed results:
  - `success`: Boolean indicating completion
  - `output`: Actual stdout/stderr from execution
  - `error`: Error message if failed
  - `metadata`: Execution time, file paths, etc.
- Results included in Brain's context for next turn
- Enables reality-based decision making

## Architecture Components

### GoalTracker (`orchestrator/goal_tracker.py`)
Manages persistent goals and subtasks in Supabase.

```python
# Create a goal
goal = await goal_tracker.create_goal(
    user_id="user123",
    description="Create a Flask todo app with SQLite",
    priority=8,
    metadata={"domain": "code", "complexity": "medium"}
)

# Add subtasks
subtasks = [
    SubTask(
        subtask_id="uuid1",
        goal_id=goal.goal_id,
        description="Install Flask and SQLAlchemy",
        action_type="aether-install",
        action_params={"packages": ["flask", "sqlalchemy"]},
        dependencies=[]
    ),
    SubTask(
        subtask_id="uuid2",
        goal_id=goal.goal_id,
        description="Create app.py with routes",
        action_type="aether-write",
        action_params={"path": "app.py", "content": "..."},
        dependencies=["uuid1"]  # Depends on packages being installed
    )
]
await goal_tracker.add_subtasks(goal.goal_id, subtasks)

# Update status after execution
await goal_tracker.update_subtask_status(
    goal.goal_id,
    subtask_id="uuid1",
    status=TaskStatus.COMPLETED,
    execution_result={"success": True, "output": "Flask installed"}
)
```

### AutonomousAgent (`orchestrator/autonomous_agent.py`)
Self-healing agent that works until goals complete.

```python
# Agent automatically:
# 1. Decomposes goals into subtasks using Brain
# 2. Executes subtasks respecting dependencies
# 3. Analyzes failures and generates fixes
# 4. Retries with corrections
# 5. Continues until completion or max attempts

success = await autonomous_agent.work_on_goal(goal)
```

**Key Methods:**
- `work_on_goal(goal)`: Main loop, executes until complete
- `_execute_subtask_with_healing(goal, subtask)`: Execute with retry logic
- `_generate_fix_strategy(subtask, failure_result)`: Brain analyzes errors
- `decompose_goal_into_subtasks(goal)`: Brain creates execution plan

### BackgroundWorker (`orchestrator/background_worker.py`)
Continuously processes goals in background.

```python
# Runs on server startup
worker = BackgroundWorker(
    brain=BRAIN,
    heart=HEART,
    store=STORE,
    memory=MEMORY,
    action_parser=ACTION_PARSER,
    router=ROUTER,
    poll_interval=30  # Check every 30 seconds
)

await worker.start()  # Runs forever, processes pending goals
```

**Workflow:**
1. Poll database for pending goals
2. Decompose goals without subtasks
3. Spawn autonomous agent for each goal
4. Agent works independently until completion
5. Results stored in database

### Enhanced ActionExecutor
Returns detailed execution results:

```python
result = await action_executor.execute(action_tag, user_id)
# Returns:
{
    "success": True,
    "result": "File written successfully",
    "output": "Created app.py (1500 bytes)",
    "error": None,
    "metadata": {
        "execution_time": 0.05,
        "file_path": "/path/to/app.py",
        "file_size": 1500
    }
}
```

### Active Inference Feedback Loop
Execution results feed back to Brain:

```python
# In active_inference.py:
# 1. Execute actions, collect results
execution_results = []
for action_tag in action_tags:
    result = await action_executor.execute(action_tag, user_id)
    execution_results.append(result)

# 2. Store for next turn
session_data["last_execution_results"] = execution_results

# 3. Next turn: Include in Brain's context
if last_execution_results:
    feedback_str = "\n\n## EXECUTION RESULTS FROM PREVIOUS TURN:\n"
    for result in last_execution_results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        feedback_str += f"\n{result['action_type']}: {status}\n"
        if result.get("error"):
            feedback_str += f"Error: {result['error']}\n"
    combined_context += feedback_str
```

Brain sees actual results and can:
- Retry failed operations with fixes
- Verify success before proceeding
- Adjust strategy based on real outcomes

## API Endpoints

### Create Autonomous Goal
```bash
POST /v1/goals/create
Authorization: Bearer <api_key>

{
    "description": "Create a Flask todo app with SQLite database",
    "priority": 8,
    "metadata": {"domain": "code", "complexity": "medium"}
}

# Response:
{
    "goal_id": "uuid",
    "status": "pending",
    "message": "Goal submitted. It will be processed autonomously in the background.",
    "check_status_url": "/v1/goals/uuid/status"
}
```

### Check Goal Status
```bash
GET /v1/goals/{goal_id}/status
Authorization: Bearer <api_key>

# Response:
{
    "goal_id": "uuid",
    "status": "in_progress",
    "description": "Create a Flask todo app...",
    "progress": {
        "completed": 2,
        "failed": 0,
        "total": 5,
        "percentage": 40.0
    },
    "subtasks": [
        {
            "subtask_id": "uuid1",
            "description": "Install Flask",
            "status": "completed",
            "attempt_count": 1,
            "execution_result": {
                "success": true,
                "output": "Flask-3.0.0 installed"
            }
        },
        {
            "subtask_id": "uuid2",
            "description": "Create app.py",
            "status": "in_progress",
            "attempt_count": 1
        }
    ],
    "created_at": "2026-01-05T19:00:00Z",
    "updated_at": "2026-01-05T19:02:30Z"
}
```

### List User Goals
```bash
GET /v1/goals/list?status_filter=in_progress
Authorization: Bearer <api_key>

# Response:
{
    "goals": [
        {
            "goal_id": "uuid1",
            "description": "Create Flask todo app",
            "status": "in_progress",
            "priority": 8,
            "progress": {"completed": 2, "total": 5, "percentage": 40.0}
        }
    ],
    "total": 1
}
```

## Database Schema

### Supabase Goals Table
```sql
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

CREATE INDEX idx_goals_user_status ON goals(user_id, status);
CREATE INDEX idx_goals_priority ON goals(priority DESC);
```

Run schema migration:
```bash
psql <supabase_connection_string> < scripts/supabase_goals_schema.sql
```

## Brain System Prompts

Updated prompts emphasize:
1. **Reality-based execution** - check actual results
2. **Step-by-step verification** - verify each step before proceeding
3. **No hallucination** - never assume success
4. **Error recovery** - diagnose and fix failures
5. **Continuous work** - keep going until task complete

Example thinking structure:
```xml
<think>
• **Plan implementation with verification points**
  - **Step 1**: Install Flask
    → VERIFY: Check installation succeeded before Step 2
  - **Step 2**: Create app.py
    → VERIFY: File actually written to disk
  - **Step 3**: Test application
    → VERIFY: Server starts without errors

• **Error recovery strategy**
  - If installation fails → Check error message, try alternative
  - If file creation fails → Check permissions, retry
  - **NEVER proceed if previous step failed** - Fix it first!
</think>
```

## Usage Examples

### Example 1: Simple Task
```python
# User asks: "Create a Python calculator script"

# AetherMind:
# 1. Creates goal in database
# 2. Decomposes into subtasks:
#    - Write calculator.py
#    - Test basic operations
# 3. Executes subtasks
# 4. If test fails, reads error, fixes code, retries
# 5. Continues until all tests pass
```

### Example 2: Complex App
```python
# User asks: "Build a Flask bookstore app with templates"

# AetherMind:
# 1. Creates goal with priority 8
# 2. Brain decomposes into subtasks:
#    a. Install Flask
#    b. Create app.py with routes
#    c. Create templates/index.html
#    d. Create templates/book.html
#    e. Create static/style.css
#    f. Test application
# 3. Executes in order, respecting dependencies
# 4. If Flask install fails:
#    - Brain sees error: "pip: command not found"
#    - Generates fix: Use python -m pip instead
#    - Retries with corrected command
# 5. If file creation fails:
#    - Brain sees error: "Permission denied"
#    - Generates fix: Use ~/AetherMind_Workspace/
#    - Retries with corrected path
# 6. Continues until app runs successfully
```

### Example 3: Resume After Restart
```python
# Scenario: Server crashes during execution

# Before crash:
# - Goal: "Create Flask app"
# - Status: in_progress
# - Completed subtasks: 2/5
# - Pending subtasks: 3

# After restart:
# 1. Background worker starts
# 2. Polls database, finds in_progress goal
# 3. Spawns autonomous agent
# 4. Agent continues from where it left off
# 5. Executes remaining 3 subtasks
# 6. Marks goal as completed
```

## Configuration

### Enable Background Worker
Already enabled in `main_api.py`:
```python
BACKGROUND_WORKER = BackgroundWorker(
    brain=BRAIN,
    heart=HEART,
    store=STORE,
    memory=MEMORY,
    action_parser=ACTION_PARSER,
    router=ROUTER,
    poll_interval=30  # Adjust polling frequency
)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(BACKGROUND_WORKER.start())
```

### Adjust Retry Limits
In `orchestrator/goal_tracker.py`:
```python
SubTask(
    ...
    max_attempts=3,  # Change retry limit (default 3)
    ...
)
```

### Change Polling Interval
In `main_api.py`:
```python
BACKGROUND_WORKER = BackgroundWorker(
    ...
    poll_interval=60  # Poll every 60 seconds instead of 30
)
```

## Monitoring & Debugging

### Logs
All components use structured logging:
```
2026-01-05 19:00:00 | INFO | 🎯 Starting autonomous work on goal: Create Flask app
2026-01-05 19:00:05 | INFO | 📊 Goal progress: 1/5 (20.0%)
2026-01-05 19:00:10 | INFO | 🔧 Executing subtask: Install Flask
2026-01-05 19:00:15 | SUCCESS | ✅ Subtask completed: Install Flask
2026-01-05 19:00:20 | WARNING | ⚠️ Subtask failed: name 'os' is not defined
2026-01-05 19:00:25 | INFO | 🧠 Brain diagnosis: Missing import statement
2026-01-05 19:00:30 | INFO | 🔄 Attempting fix: Add import os at top of file
2026-01-05 19:00:35 | SUCCESS | ✅ Subtask completed after retry
```

### Check Active Goals
```bash
curl -H "Authorization: Bearer <api_key>" \
  http://localhost:8000/v1/goals/list
```

### Monitor Goal Progress
```bash
watch -n 5 "curl -s -H 'Authorization: Bearer <api_key>' \
  http://localhost:8000/v1/goals/<goal_id>/status | jq '.progress'"
```

## Best Practices

1. **Break Down Complex Tasks**: Brain will decompose, but clearer requests help
   - Good: "Create a Flask todo app with SQLite database and user auth"
   - Better: "Create a Flask todo app. Include SQLite for data storage, user authentication with bcrypt, and CRUD operations for tasks."

2. **Set Appropriate Priorities**: Use 1-10 scale
   - 1-3: Low priority (experiments, nice-to-haves)
   - 4-6: Medium priority (normal tasks)
   - 7-9: High priority (urgent work)
   - 10: Critical (only for essential tasks)

3. **Include Domain Metadata**: Helps Brain choose right approach
   ```json
   {
       "metadata": {
           "domain": "code",
           "language": "python",
           "complexity": "medium",
           "testing_required": true
       }
   }
   ```

4. **Monitor Long-Running Goals**: Check status periodically
   ```bash
   # Set up monitoring
   watch -n 30 "curl -s ... | jq '.progress.percentage'"
   ```

5. **Trust the Self-Healing**: Don't manually intervene unless goal fails completely
   - Agent will retry failed steps automatically
   - Brain learns from errors and adjusts approach
   - Only intervene if status is "failed" with max attempts reached

## Limitations & Future Work

### Current Limitations
1. **Max Iterations**: Goals timeout after 50 reasoning cycles (prevents infinite loops)
2. **Max Retries**: Subtasks fail after 3 attempts (configurable)
3. **No Concurrent Subtasks**: Executes one at a time (could parallelize independent tasks)
4. **Limited Error Analysis**: Brain's diagnostic capabilities depend on prompt quality

### Planned Enhancements
1. **Parallel Execution**: Run independent subtasks concurrently
2. **Learning from Failures**: Store error patterns in vector DB for future reference
3. **User Notifications**: Webhook/email when goals complete or fail
4. **Resource Limits**: CPU/memory constraints for sandboxes
5. **Cost Tracking**: Monitor API usage per goal
6. **Progress Streaming**: WebSocket updates for real-time progress

## Troubleshooting

### Background Worker Not Starting
**Symptom**: Goals stay in "pending" status forever

**Solution**:
```bash
# Check logs for startup errors
tail -f logs/aethermind.log | grep BackgroundWorker

# Verify Redis connection (required for worker)
redis-cli ping

# Restart server
uvicorn orchestrator.main_api:app --reload --port 8000
```

### Goals Failing Immediately
**Symptom**: All goals fail with "Could not decompose into subtasks"

**Solution**:
- Check Brain model is accessible
- Verify GOOGLE_API_KEY is set
- Test Brain directly:
  ```python
  response = await brain.generate_thought("Create a Flask app", ...)
  print(response)
  ```

### Subtasks Stuck in "retrying"
**Symptom**: Subtasks retry forever without completing

**Solution**:
- Check logs for actual error messages
- Verify action executors (PracticeAdapter, etc.) are working
- Test action execution directly:
  ```python
  result = await action_executor.execute(action_tag, user_id)
  print(result)
  ```

### Database Connection Issues
**Symptom**: "Failed to create goal" errors

**Solution**:
```bash
# Verify Supabase credentials
echo $SUPABASE_URL
echo $SUPABASE_ANON_KEY

# Test Supabase connection
python -c "from orchestrator.supabase_client import get_supabase_client; print(get_supabase_client())"

# Run schema migration
psql <connection_string> < scripts/supabase_goals_schema.sql
```

## Summary

AetherMind now has full AGI-level capabilities:
- ✅ Autonomous task completion
- ✅ Self-healing error recovery
- ✅ Persistent goal tracking
- ✅ Background execution
- ✅ Multi-turn reasoning
- ✅ Reality-based decision making

The system can independently complete complex tasks, recover from failures, and continue working even when users are offline. Execution results feed back to the Brain, enabling continuous learning and adjustment until goals are achieved.
