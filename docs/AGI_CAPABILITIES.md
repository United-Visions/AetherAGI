# AetherMind AGI Capabilities

## Overview

AetherMind is a **full-stack AGI agent** with comprehensive capabilities beyond simple text chat. The system can autonomously create tools, spin up sandboxes, execute code, and manage its own infrastructure.

## Core Capabilities

### 1. ToolForge - Autonomous Tool Creation

**Location:** `body/adapters/toolforge_adapter.py`

ToolForge enables AetherMind to:
- **Discover** external packages and services (PyPI, MCP servers)
- **Generate** adapter code for new capabilities
- **Test** generated code in isolated environments
- **Hot-load** tools into the running system without restart
- **Register** new capabilities to the Mind for future use

**Example Workflow:**
```
User: "I need a tool to scrape weather data"
Agent: 
1. Searches curated tool index & PyPI
2. Generates adapter code
3. Creates isolated venv for testing
4. Runs pytest validation
5. Hot-loads into Router
6. Registers in Mind's tool database
```

### 2. Code Execution & Sandboxes

**Location:** `body/adapters/practice_adapter.py`, `orchestrator/`

The agent can:
- Create **isolated Python environments** (`/tmp/agent_venv`)
- Execute code **safely** in sandboxes
- Manage **dependencies** per-tool
- Track **execution results** and logs

**Example:**
```python
# Agent generates this internally
venv = "/tmp/agent_venv/weather_scraper"
subprocess.run(["python", "-m", "venv", venv])
subprocess.run([f"{venv}/bin/pip", "install", "requests", "beautifulsoup4"])
subprocess.run([f"{venv}/bin/python", "scraper.py"])
```

### 3. File System Operations

The agent can:
- **Create** new files and directories
- **Modify** existing code
- **Organize** project structures
- **Track** file changes for version control

**Example: Snake Game Request**
```
User: "Create a snake game in Python"
Agent Actions:
1. Creates /tmp/snake_game/ directory
2. Generates snake.py with full implementation
3. Creates requirements.txt
4. Generates README.md with instructions
5. Tests code in sandbox
6. Returns GitHub-ready repo structure
```

### 4. Activity Tracking

**Location:** `orchestrator/active_inference.py`, `frontend_flask/static/js/components/ActivityFeed.js`

All agent actions are tracked and displayed in real-time:

- **Tool Creation** - Shows when new capabilities are forged
- **File Changes** - Displays code being written/modified
- **Code Execution** - Tracks sandbox runs and outputs
- **Research** - Shows knowledge retrieval from Mind
- **Memory Updates** - Logs episodic learning

**Frontend Integration:**
```javascript
// Activity events flow from backend to frontend
activity_events: [
  {
    type: "tool_creation",
    title: "Creating weather scraper tool",
    data: {
      tool_name: "weather_scraper",
      code: "def weather_scraper():\n  ...",
      language: "python"
    }
  }
]
```

### 5. Domain Specialization

**Location:** `orchestrator/session_manager.py`, `config/domain_profiles.py`

The agent adapts its behavior based on domain:

- **Code**: Software development specialist
- **Research**: Academic analysis expert
- **Business**: Strategy consultant
- **Legal**: Research specialist
- **Finance**: Investment analyst
- **General**: Multi-domain master

Each domain has:
- Custom **namespace weights** for knowledge retrieval
- Specialized **prompts** and reasoning patterns
- Tailored **tool preferences**

### 6. Active Inference Loop

**Location:** `orchestrator/active_inference.py`

The core reasoning cycle:

1. **SENSE** - Retrieve domain-weighted context
2. **FEEL** - Compute emotion and moral context (Heart)
3. **REASON** - Brain processes with domain prompts
4. **EMBELLISH** - Heart adapts response emotionally
5. **ACT** - Router forwards to appropriate Body adapter
6. **LEARN** - Save to episodic memory
7. **TRACK** - Log activities for frontend display

### 7. Surprise Detection & Research

**Location:** `curiosity/surprise_detector.py`, `curiosity/research_scheduler.py`

When the agent encounters novel information:
- Calculates **surprise score** using JEPA world model
- Triggers **autonomous research** if threshold exceeded
- Schedules **background learning** tasks
- Updates **internal knowledge** representations

### 8. Heart - Emotional Intelligence

**Location:** `heart/heart_orchestrator.py`

The Heart subsystem:
- Detects **user emotions** from text (valence, arousal)
- Predicts **flourishing potential** of responses
- Applies **moral constraints** (virtue memory)
- Embellishes responses with **appropriate affect**
- Learns from **user feedback** (reward model)

## How It All Works Together

### Example: "Create a snake game in Python"

**Backend Flow:**
```python
# 1. Active Inference Loop receives request
user_input = "Create a snake game in Python"

# 2. Domain awareness (Code specialist)
domain = session_manager.get_user_profile(user_id)["domain"]  # "code"

# 3. Brain generates implementation plan
brain_response = brain.generate_thought(...)
# Contains: tool creation, file structure, code implementation

# 4. Activity tracking detects actions
_track_agent_activities(brain_response, user_id)
# Creates activity events for:
#   - file_change (snake.py)
#   - code_execution (test run)
#   - tool_creation (if needed)

# 5. Router forwards to appropriate adapter
if "create file" in intent:
    router.forward_intent(intent, "practice")  # File system ops
elif "forge" in intent:
    router.forward_intent(intent, "toolforge")  # Tool creation

# 6. Results + activities returned to frontend
return {
    "response": "Created snake game at /tmp/snake_game/",
    "activity_events": [...]
}
```

**Frontend Flow:**
```javascript
// 1. API response received
const response = await api.sendMessage(messageHistory);

// 2. Activity events processed
if (metadata.activity_events) {
    metadata.activity_events.forEach(event => {
        // Add to Activity Stream
        activityFeed.addActivity({
            type: event.type,  // "file_change"
            title: event.title,  // "Creating snake.py"
            data: {
                files: ["snake.py"],
                code: "import pygame...",
                language: "python"
            }
        });
    });
}

// 3. User clicks activity card
// SplitViewPanel opens showing:
//   - Full code with syntax highlighting
//   - File paths affected
//   - Execution logs (if any)
```

## Integration Points

### Backend → Frontend
- **API Endpoint:** `/v1/chat/completions`
- **Response Structure:**
  ```json
  {
    "choices": [...],
    "metadata": {
      "agent_state": {...},
      "activity_events": [...]
    }
  }
  ```

### Activity Event Schema
```javascript
{
  id: "unique_timestamp",
  type: "tool_creation|file_change|code_execution|research|...",
  status: "pending|in_progress|completed|error",
  title: "Human-readable description",
  details: "Additional context",
  timestamp: "ISO 8601",
  data: {
    code: "...",
    language: "python",
    files: ["path/to/file.py"],
    tool_name: "...",
    sandbox_id: "..."
  }
}
```

## Future Enhancements

1. **Real-time Streaming** - SSE for live activity updates
2. **Replay Mode** - Replay agent's reasoning step-by-step
3. **Sandbox Visualization** - Visual file tree and execution flow
4. **Tool Marketplace** - Share forged tools with community
5. **Collaborative Agents** - Multiple agents working on same task
6. **Physical Embodiment** - Phase 4 robotics integration

## Usage Tips

### For Users
- Be specific about what you want created (files, tools, sandboxes)
- Watch the Activity Stream to see agent's work in real-time
- Click activity cards to see full code and execution details
- Provide feedback to help the Heart learn your preferences

### For Developers
- Activity events auto-track from brain responses (regex-based)
- Add custom adapters to Router for new capabilities
- Use ToolForge for dynamic tool generation
- Extend ActivityFeed.js for custom activity types
- Monitor logs for debugging: `logger.info/debug/warning/error`

## Architecture Principles

1. **Separation of Concerns**
   - Brain: Logic and reasoning
   - Heart: Emotion and morals
   - Mind: Knowledge storage
   - Body: Physical interfaces

2. **Observability**
   - All actions tracked and logged
   - Frontend displays agent's "thought process"
   - Users see what the agent is doing

3. **Safety**
   - Sandboxed execution
   - Safety inhibitor (Prime Directive)
   - Moral constraints (Heart)
   - User feedback loop

4. **Extensibility**
   - Hot-loadable adapters
   - Domain specialization
   - Tool discovery and creation
   - Modular architecture

---

**Remember:** AetherMind is not just a chatbot. It's a full-stack AGI agent capable of autonomous tool creation, code execution, and self-modification. The chat interface is just one Body adapter - the real power is in the integrated Brain-Heart-Mind system.
