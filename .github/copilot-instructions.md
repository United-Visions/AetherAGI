# AetherMind AI Coding Agent Instructions

## Project Overview

AetherMind is a **Developmental Continual Learning Architecture (DCLA)** AGI system with a biologically-inspired "digital organism" design. This is NOT a standard LLM wrapper—it's a split-brain cognitive architecture with distinct layers.

## Architecture: The Digital Organism

```
Brain (Fixed Logic) ←→ Orchestrator (Nervous System) ←→ Mind (Expandable Knowledge)
                              ↓
                    Heart (Emotion/Ethics)
                              ↓
                    Body (Interface Adapters)
```

### Core Layers

| Layer | Directory | Purpose | Key Files |
|-------|-----------|---------|-----------|
| **Brain** | `brain/` | Fixed reasoning engine (HOW to think) | `logic_engine.py`, `safety_inhibitor.py`, `system_prompts.py` |
| **Mind** | `mind/` | Infinite memory (WHAT to think about) | `vector_store.py`, `episodic_memory.py`, `promoter.py` |
| **Heart** | `heart/` | Emotion & ethics | `heart_orchestrator.py`, `reward_model.py`, `virtue_memory.py` |
| **Body** | `body/adapters/` | Output interfaces | `chat_ui.py`, `practice_adapter.py`, `toolforge_adapter.py` |
| **Orchestrator** | `orchestrator/` | Nervous system | `main_api.py`, `active_inference.py`, `action_parser.py` |

## Development Commands

### Starting the Application

**Backend (FastAPI on port 8000):**
```bash
./start_backend.sh
# OR manually:
source .venv/bin/activate && uvicorn orchestrator.main_api:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (Flask/Quart on port 5000):**
```bash
./start_frontend.sh
# OR manually:
source .venv/bin/activate && cd frontend_flask && python app.py
```

**Docker (full stack with Redis/Prometheus/Grafana):**
```bash
docker-compose up
```

### Environment Setup

Required `.env` variables:
```bash
PINECONE_API_KEY=xxx           # Vector database (required)
ADMIN_SECRET=xxx               # API key generation
GITHUB_CLIENT_ID/SECRET=xxx    # OAuth flow
FERNET_KEY=xxx                 # Token encryption
REDIS_URL=redis://localhost:6379
```

## Key Patterns & Conventions

### 1. Active Inference Loop (Core Cognitive Cycle)

All reasoning flows through `orchestrator/active_inference.py`:
```
Sense → Feel → Reason → Embellish → Parse Actions → Execute → Learn
```

When modifying cognitive behavior, trace the cycle in `ActiveInferenceLoop.run_cycle()`.

### 2. Action Tag System

The Brain communicates structured actions via XML-style tags parsed by `orchestrator/action_parser.py`:

```xml
<aether-write path="file.py" language="python">code</aether-write>
<aether-sandbox language="python" test="true">code</aether-sandbox>
<aether-forge tool="name" action="generate">{spec}</aether-forge>
<aether-install packages="flask requests"></aether-install>
<aether-research query="topic" namespace="domain"></aether-research>
<think>reasoning steps</think>
```

**17 action tag types** are defined in `ActionParser.TAG_PATTERNS`. Add new capabilities by:
1. Adding pattern to `action_parser.py`
2. Implementing executor in `ActionExecutor._execute_*`
3. Updating `system_prompts.py` with usage examples

### 3. Domain Specialization

Users select domains during onboarding. Domain profiles in `config/domain_profiles.py` control:
- Communication style
- Knowledge namespace weights
- Tool preferences
- Response formats

Available domains: `code`, `research`, `business`, `legal`, `finance`, `general`

### 4. Safety Inhibitor (Non-Trainable)

`brain/safety_inhibitor.py` is a **hard-wired safety layer** that cannot be bypassed by learning. It includes:
- Text-based harm detection
- Kinetic safety for hardware (GPIO/Serial/RTSP)
- Critical pin protection lists

**Never remove or weaken safety checks** without explicit approval.

### 5. Vector Store Namespaces

Pinecone namespaces organize knowledge:
- `core_universal` - Base knowledge
- `domain_code/research/legal/etc` - Specialized domains
- `user_{id}_episodic` - Per-user conversation memory

Query pattern: `store.query_context(query, namespace="core_universal")`

### 6. Session & Learning Context

`SessionManager` tracks user state and learning:
```python
session_mgr.set_user_domain(user_id, "code")
profile = session_mgr.get_user_profile(user_id)
mega_prompt = session_mgr.get_mega_prompt_prefix(user_id)
```

### 7. LLM Integration

Uses **LiteLLM** for model abstraction in `brain/logic_engine.py`:
```python
response = await litellm.acompletion(
    model="gemini/gemini-2.5-pro",
    messages=messages,
    fallbacks=["gemini/gemini-1.5-pro", "openai/gpt-4o"]
)
```

### 8. Body Adapters

New interfaces implement `body/adapter_base.py`:
```python
class MyAdapter(BodyAdapter):
    async def execute(self, intent: str) -> str:
        spec = json.loads(intent)
        # Process and return result
```

Register in `orchestrator/router.py` and enable in `config/settings.yaml`.

### 9. Frontend Supabase Integration

When using Supabase in frontend templates (via CDN):
- The `supabase-js` library exposes a global `supabase` object.
- **DO NOT** redeclare `supabase` (e.g., `let supabase = null`). This causes `Identifier 'supabase' has already been declared` errors.
- Destructure `createClient` from the global object: `const { createClient } = supabase;`.
- Assign the client to a distinct variable (e.g., `sbClient`) to avoid shadowing conflicts.

```javascript
// Correct Pattern
const { createClient } = supabase;
const sbClient = createClient(SUPABASE_URL, SUPABASE_KEY);
```

## File Organization

```
orchestrator/main_api.py    # FastAPI entry point, all endpoints
orchestrator/active_inference.py  # Core cognitive loop
brain/system_prompts.py     # All system prompts with action tag docs
brain/safety_inhibitor.py   # Safety layer (DO NOT WEAKEN)
config/settings.yaml        # Feature flags
config/domain_profiles.py   # Domain personality configurations
frontend_flask/app.py       # Flask/Quart frontend (uses async Quart)
sdk/                        # Python & JavaScript SDKs
```

## API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | OpenAI-compatible chat |
| `POST /v1/chat` | SDK chat with domain support |
| `POST /v1/user/domain` | Set user domain specialization |
| `POST /v1/goals/create` | Autonomous background goals |
| `POST /v1/ingest/multimodal` | Image/video processing via Eye |
| `POST /v1/tools/create` | ToolForge runtime tool creation |

## Testing

```bash
# SDK endpoint tests
python sdk/test_api_endpoints.py

# Run generated tests
cd tests/agent_generated && pytest
```

## Common Tasks

### Adding a New Body Adapter
1. Create `body/adapters/my_adapter.py` extending `BodyAdapter`
2. Add to `Router.__init__()` with feature flag
3. Add flag to `config/settings.yaml`
4. Add action tag pattern if needed

### Adding a New Action Tag
1. Add regex pattern to `ActionParser.TAG_PATTERNS`
2. Implement `_extract_data()` case in `ActionTag`
3. Implement `_execute_*()` method in `ActionExecutor`
4. Document in `brain/system_prompts.py`

### Modifying Domain Behavior
Edit profiles in `config/domain_profiles.py`:
- `system_prompt` - Core personality
- `namespace_weights` - Knowledge priorities
- `tool_preferences` - Allowed tools
- `response_format` - Output structure

---

## SDK Distribution

### Python SDK (PyPI)

Located in `sdk/python/`. Package name: `aethermind`

```bash
# Build
cd sdk/python
pip install build
python -m build

# Publish to TestPyPI first
pip install twine
twine upload --repository testpypi dist/*

# Publish to production PyPI
twine upload dist/*
```

### JavaScript SDK (npm)

Located in `sdk/javascript/`. Package name: `@aethermind/sdk`

```bash
# Build
cd sdk/javascript
npm install
npm run build

# Publish
npm login
npm publish --access public
```

### SDK Core Methods

Both SDKs expose the same interface:
- `chat(message, namespace)` - Main chat endpoint
- `searchMemory(query, topK)` - Episodic memory search
- `createTool(name, code, params)` - ToolForge runtime tool creation
- `createKnowledgeCartridge(name, documents)` - Custom knowledge loading
- `getUsage()` - Rate limit and token usage

---

## Hardware Adapters (Cyber-Physical Integration)

AetherMind can control physical hardware via `body/adapters/hardware_adapter.py`.

### Supported Protocols

| Protocol | Use Case | Dependencies |
|----------|----------|--------------|
| **GPIO** | Raspberry Pi pins, sensors, actuators | `RPi.GPIO` |
| **Serial/UART** | Arduino, motor controllers, PLCs | `pyserial` |
| **RTSP** | Network cameras, surveillance | `opencv-python` |

### Hardware Intent JSON Format

```json
{
  "protocol": "GPIO|SERIAL|RTSP",
  "action": "read|write|connect|capture",
  "params": { /* protocol-specific */ },
  "metadata": { "safety_approved": true }
}
```

### Safety: Critical Pin Protection

The `SafetyInhibitor.check_kinetic_safety()` validates ALL hardware intents before execution:
- **Critical GPIO pins** (e.g., pin 21 emergency shutdown) are blacklisted
- **Dangerous serial patterns** (`EMERGENCY_OVERRIDE`, `DISABLE_SAFETY`) are blocked
- **RTSP localhost** access is forbidden

```python
# Example: Safety check flow
from brain.safety_inhibitor import SafetyInhibitor
inhibitor = SafetyInhibitor()
is_safe, message = inhibitor.check_kinetic_safety(intent_json)
if not is_safe:
    return inhibitor.kinetic_inhibition_response
```

### Additional Hardware Adapters

| Adapter | File | Purpose |
|---------|------|---------|
| `automotive.py` | Vehicle control, navigation |
| `smart_home.py` | Home automation, IoT devices |
| `vision_system.py` | YOLO object detection, visual analysis |

---

## Background Worker & Autonomous Goals

AetherMind can work autonomously in the background via `orchestrator/background_worker.py`.

### How It Works

1. **User submits goal** via `POST /v1/goals/create`
2. **GoalTracker** stores goal in Supabase with `pending` status
3. **BackgroundWorker** polls every 30 seconds for pending goals
4. **AutonomousAgent** decomposes goal into subtasks
5. **Each subtask** executes via action tags, with retry on failure
6. **Progress tracked** in real-time, survives server restarts

### Goal Lifecycle

```
pending → in_progress → [retrying] → completed|failed
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `BackgroundWorker` | `orchestrator/background_worker.py` | Continuous polling loop |
| `GoalTracker` | `orchestrator/goal_tracker.py` | Supabase persistence |
| `AutonomousAgent` | `orchestrator/autonomous_agent.py` | Goal decomposition & execution |

### SubTask Structure

```python
SubTask(
    subtask_id="uuid",
    goal_id="parent_uuid",
    description="Install Flask",
    action_type="aether-install",
    action_params={"packages": "flask"},
    dependencies=[],  # Other subtask IDs
    status=TaskStatus.PENDING,
    max_attempts=3
)
```

### Running Standalone Worker

```bash
python -m orchestrator.background_worker
```

---

## Curiosity & Surprise Detection

AetherMind autonomously researches novel concepts via `curiosity/surprise_detector.py`.

### Surprise Scoring Formula

```
surprise = (0.7 × JEPA_energy) + (0.3 × novelty)
```

- **JEPA energy**: Prediction error from world model (`brain/jepa_aligner.py`)
- **Novelty**: 1 - max cosine similarity to `autonomous_research` namespace

### Research Trigger Flow

1. User input generates embedding vector
2. `SurpriseDetector.score(vector)` calculates surprise
3. If `surprise > 0.5` (threshold), research job is queued
4. `ResearchScheduler` pushes to Redis sorted set (priority = surprise)
5. Background worker pops and executes research

### Redis Queue Structure

```python
# Push high-priority research job
await research_scheduler.push({
    "query": "quantum computing applications",
    "surprise": 0.78,
    "tools": ["browser", "arxiv"],
    "deadline": "2026-01-07T00:00:00",
    "user_id": "github_user"
})

# Pop highest-priority job
job = await research_scheduler.pop()
```

### 24-Hour Cache

Surprise detector caches seen vectors for 24 hours to prevent duplicate research.

---

## Authentication Flow

### API Key Format

User keys follow pattern: `am_live_XXXXX` (32-char URL-safe token)

### Key Generation Flow

1. User authenticates via GitHub OAuth (`/github_login`)
2. Callback exchanges code for GitHub access token
3. Token encrypted with Fernet and stored in session
4. `AuthManagerSupabase.generate_api_key()` creates hashed key
5. Key stored in Supabase `api_keys` table
6. Plaintext key returned to user ONCE (never stored)

### Key Verification

```python
# In main_api.py
user_data = AUTH.verify_api_key(api_key)
# Returns: {"user_id": str, "role": str, "permissions": [...]}
```

### Role-Based Access Control (RBAC)

| Role | Rate Limit | Permissions |
|------|------------|-------------|
| `free` | 100/min | read, write |
| `pro` | 1000/min | + delete, meta_controller, self_modify, tool_forge |
| `enterprise` | 10000/min | + view_audit, manage_keys |
| `admin` | ∞ | All permissions |

### Supabase Tables

```sql
-- api_keys table
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    github_username TEXT,
    github_url TEXT,
    key_hash TEXT UNIQUE,  -- SHA-256 of plaintext key
    role TEXT DEFAULT 'pro',
    metadata JSONB,
    created_at TIMESTAMP,
    last_used TIMESTAMP,
    revoked BOOLEAN DEFAULT false
);
```

### GitHub OAuth Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/github_login` | Initiates OAuth redirect |
| `/callback` | Exchanges code for token |
| `/onboarding` | Domain selection after auth |
| `/create_key` | Generates API key |

---

## External Dependencies

| Service | Purpose | Config Key |
|---------|---------|------------|
| Pinecone | Vector DB | `PINECONE_API_KEY` |
| Redis | Background jobs, research queue | `REDIS_URL` |
| LiteLLM | LLM routing with fallbacks | Model names in code |
| Supabase | User auth, goal storage | `SUPABASE_URL`, `SUPABASE_KEY` |
| GitHub | OAuth authentication | `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` |

## Critical Warnings

1. **SafetyInhibitor is sacred** - Never bypass or weaken `brain/safety_inhibitor.py`
2. **Environment loading order** - `load_dotenv()` must run BEFORE importing components
3. **Async everywhere** - Frontend uses Quart (async Flask), backend is FastAPI
4. **Action tags parse strictly** - Malformed tags are silently dropped
5. **Namespace isolation** - User data uses `user_{id}_*` namespaces
6. **Kinetic safety is non-negotiable** - All hardware intents must pass `check_kinetic_safety()`
7. **Background worker is stateful** - Goals persist across restarts via Supabase

## Architecture Principles

1. **Split-Brain Design**: Brain (fixed logic) ≠ Mind (expandable knowledge)
2. **Active Inference**: Minimize "surprise" via prediction error
3. **Safety First**: Non-trainable safety layers for text AND hardware
4. **Infinite Memory**: No context window limits via Pinecone
5. **Modular Body**: Same Brain works in chat, IDE, robotics
6. **Curiosity-Driven**: Surprise detection triggers autonomous research
7. **Goal Persistence**: User goals complete even if browser closes
8. **Self-Healing**: Failed subtasks retry automatically with error context
