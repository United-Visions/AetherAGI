# AetherMind Sandbox Lifecycle

## When and Why Sandboxes Start

### Overview
Sandboxes in AetherMind are **isolated Python virtual environments** that provide safe execution contexts for:
- Testing newly generated tools
- Running user-requested code
- Installing and validating external packages
- Preventing system contamination

**Location:** `/tmp/agent_venv/`

---

## Trigger Points (When Sandboxes Start)

### 1. **ToolForge Testing** → `_test()` method
**File:** `body/adapters/toolforge_adapter.py:95`

**Trigger:** When Brain decides to create and test a new tool

```python
# User asks: "Create a weather scraping tool"
# Brain response includes: "create tool" or "forge"

# ToolForge flow:
def _test(self, name: str) -> str:
    # 1. Create isolated venv
    venv = f"{VENV_BASE}/venv_{name}"
    subprocess.run(["python", "-m", "venv", venv], check=True)
    
    # 2. Install test dependencies
    pip = f"{venv}/bin/pip"
    subprocess.run([pip, "install", "pytest", "httpx"], capture_output=True)
    
    # 3. Generate test file
    test_path = f"{VENV_BASE}/test_{name}.py"
    
    # 4. Run tests in isolation
    py = f"{venv}/bin/python"
    subprocess.run([py, "-m", "pytest", test_path, "-x"])
```

**Why:** Ensures the generated tool code works before hot-loading into production system.

---

### 2. **PyPI Package Installation** → `_pypi_install()` method
**File:** `body/adapters/toolforge_adapter.py:87`

**Trigger:** When Brain needs an external package

```python
# User asks: "Install the requests library"
# Brain decides to use ToolForge pypi_install action

def _pypi_install(self, pkg: str) -> str:
    # 1. Create package-specific venv
    venv = f"{VENV_BASE}/pypi_{pkg}"
    subprocess.run(["python", "-m", "venv", venv], check=True)
    
    # 2. Install package in isolation
    pip = f"{venv}/bin/pip"
    subprocess.run([pip, "install", pkg], check=True)
    
    return f"installed {pkg} into {venv}"
```

**Why:** Prevents dependency conflicts with the main system.

---

### 3. **Code Execution** → `PracticeAdapter._run_python()` method
**File:** `body/adapters/practice_adapter.py:26`

**Trigger:** When Brain needs to execute user code or test generated code

```python
# User asks: "Run this Python script"
# Brain forwards to PracticeAdapter

async def _run_python(self, code: str, tests: list) -> str:
    # 1. Create temporary file (lightweight sandbox)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n# --- tests ---\n")
        for t in tests:
            f.write(t + "\n")
    
    # 2. Execute in subprocess (isolated from main process)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, f.name,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 3. Capture output safely
    stdout, stderr = await proc.communicate()
    
    # 4. Clean up
    os.unlink(f.name)
```

**Why:** Prevents malicious/buggy code from crashing the main agent process.

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ USER REQUEST                                                     │
│ "Create a web scraper" / "Install pandas" / "Run this code"     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR: Active Inference Loop                              │
│ - Sense: Retrieve context                                        │
│ - Reason: Brain generates plan                                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ BRAIN: Logic Engine                                              │
│ Response includes patterns like:                                 │
│ - "create tool weather_scraper"                                  │
│ - "install package pandas"                                       │
│ - "execute this code"                                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ ROUTER: Determines which Body adapter to use                     │
│ - Contains "forge" → ToolForgeAdapter                            │
│ - Contains "execute"/"run" → PracticeAdapter                     │
│ - Default → ChatAdapter                                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ BODY ADAPTER EXECUTES                                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌─────────┐ ┌────────────┐
│ ToolForge    │ │Practice │ │  Chat      │
│              │ │         │ │            │
│ Actions:     │ │Actions: │ │Actions:    │
│ • discover   │ │• run_py │ │• format    │
│ • generate   │ │• run_sh │ │  output    │
│ • TEST ←─────┼─┼─────────┼─┼────────────┼── SANDBOX CREATED
│ • load       │ │         │ │            │
│ • pypi_inst  │ │         │ │            │
└──────────────┘ └─────────┘ └────────────┘
       │              │
       │              │
       ▼              ▼
┌─────────────────────────────────────────┐
│ SANDBOX ENVIRONMENT                      │
│ /tmp/agent_venv/venv_{tool_name}/       │
│ /tmp/agent_venv/pypi_{package}/         │
│ /tmp/{temp_file}.py                     │
│                                          │
│ Structure:                               │
│ ├── bin/                                 │
│ │   ├── python                           │
│ │   ├── pip                              │
│ │   └── pytest                           │
│ ├── lib/                                 │
│ │   └── python3.x/                       │
│ │       └── site-packages/               │
│ └── test_{tool_name}.py                  │
└─────────────────────────────────────────┘
```

---

## Detailed Execution Path Examples

### Example 1: User Requests "Create a snake game"

```
1. USER INPUT
   └─> "Create a snake game in Python"

2. ACTIVE INFERENCE LOOP
   └─> Retrieves context about Python games, pygame
   └─> _track_agent_activities() detects: "create", "game"

3. BRAIN RESPONSE
   └─> "I will create a snake game. Let me generate the code..."
   └─> Includes: ```python [snake game code] ```

4. ACTIVITY TRACKING
   └─> Detects pattern: "create file"
   └─> Creates activity_event: type="file_change"

5. ROUTER DECISION
   └─> Sees "execute" or "test" keywords
   └─> Routes to PracticeAdapter OR ToolForge

6. SANDBOX CREATION (if testing)
   └─> venv = "/tmp/agent_venv/venv_snake_game"
   └─> subprocess.run(["python", "-m", "venv", venv])
   └─> Writes snake.py to temp location
   └─> Runs: "/tmp/agent_venv/venv_snake_game/bin/python snake.py"

7. RESULT RETURNED
   └─> stdout/stderr captured
   └─> Activity updated: status="completed"
   └─> User sees: "Snake game created and tested successfully"
```

### Example 2: User Requests "Install beautifulsoup4"

```
1. USER INPUT
   └─> "Install beautifulsoup4 for web scraping"

2. BRAIN RESPONSE
   └─> "I'll install beautifulsoup4 via ToolForge"
   └─> Intent: {"action": "pypi_install", "name": "beautifulsoup4"}

3. ROUTER
   └─> Routes to ToolForgeAdapter

4. SANDBOX CREATION (pypi_install)
   └─> venv = "/tmp/agent_venv/pypi_beautifulsoup4"
   └─> subprocess.run(["python", "-m", "venv", venv])
   └─> pip = "/tmp/agent_venv/pypi_beautifulsoup4/bin/pip"
   └─> subprocess.run([pip, "install", "beautifulsoup4"])

5. ADAPTER GENERATION
   └─> Generates pypi_beautifulsoup4_adapter.py
   └─> Wraps package in standardized interface

6. TESTING (optional)
   └─> Creates test venv
   └─> Runs smoke tests
   └─> Validates import works

7. HOT-LOADING
   └─> Loads adapter into ROUTER.adapters{}
   └─> Agent can now use beautifulsoup4 in future requests
```

---

## Why Separate Sandboxes?

### Security
- **Process Isolation**: Code runs in subprocess, can't crash main agent
- **File System**: Limited access to /tmp, not main system
- **Memory**: Each sandbox has own memory space
- **Network**: Can be restricted with additional policies

### Dependency Management
- **No Conflicts**: Each tool has its own dependency tree
- **Version Control**: Different tools can use different package versions
- **Clean State**: Each execution starts fresh

### Testing & Validation
- **Pre-Production**: Test before hot-loading into production
- **Rollback**: If tests fail, nothing is loaded
- **Audit Trail**: All sandbox operations logged

---

## Sandbox Lifecycle States

```
┌──────────────┐
│  DORMANT     │  No sandbox exists
└──────┬───────┘
       │ (Trigger: Tool test, PyPI install, Code execution)
       ▼
┌──────────────┐
│  CREATING    │  python -m venv {path}
└──────┬───────┘  Creating directory structure
       │
       ▼
┌──────────────┐
│  INSTALLING  │  pip install {packages}
└──────┬───────┘  Installing dependencies
       │
       ▼
┌──────────────┐
│  TESTING     │  pytest test_{name}.py
└──────┬───────┘  Running validation
       │
       ├─ SUCCESS → HOT-LOAD → ACTIVE
       │
       └─ FAILURE → CLEANUP → DORMANT
                    (Delete venv)
```

---

## Key Configuration

### Environment Variables
```bash
# Where sandboxes are created
VENV_BASE="/tmp/agent_venv"

# Tool index for discovery
TOOL_INDEX_PATH="/app/curated_tool_index.json"
```

### Settings
```yaml
# config/settings.yaml
toolforge_adapter: true    # Enable/disable ToolForge
practice_adapter: true     # Enable/disable PracticeAdapter
```

---

## Monitoring Sandboxes

### Check Active Sandboxes
```bash
ls -la /tmp/agent_venv/
# Output:
# venv_weather_scraper/
# pypi_requests/
# pypi_beautifulsoup4/
# test_weather_scraper.py
```

### Inspect Sandbox Contents
```bash
ls -la /tmp/agent_venv/pypi_requests/
# bin/     - Python interpreter, pip, etc.
# lib/     - Installed packages
# include/ - Header files
```

### View Test Results
```bash
cat /tmp/agent_venv/test_weather_scraper.py
# Shows generated test code
```

---

## Cleanup Policy

Currently: **Manual cleanup required**

Sandboxes persist in `/tmp/` until:
1. System reboot (automatic /tmp cleanup)
2. Manual deletion
3. Future: Implement TTL-based cleanup service

**Recommended Enhancement:**
```python
# Future: Add to orchestrator
class SandboxManager:
    def cleanup_old_sandboxes(self, max_age_hours=24):
        """Remove sandboxes older than max_age_hours"""
        for venv in Path(VENV_BASE).glob("*"):
            if venv.stat().st_mtime < (time.time() - max_age_hours * 3600):
                shutil.rmtree(venv)
                logger.info(f"Cleaned up old sandbox: {venv}")
```

---

## Summary

**When:** Sandboxes start when the Brain decides to:
1. Test a newly generated tool (ToolForge)
2. Install an external package (ToolForge)
3. Execute user/generated code (PracticeAdapter)

**Why:** 
- **Safety**: Isolated execution prevents system corruption
- **Testing**: Validate before production deployment
- **Dependencies**: Manage per-tool package requirements
- **Reliability**: Failure in sandbox doesn't crash agent

**Where:** `/tmp/agent_venv/{venv_name}/`

**How:** Python's `venv` module + subprocess execution

The sandbox system is a critical safety and testing layer that allows AetherMind to autonomously create, test, and deploy new capabilities without risking system stability.
