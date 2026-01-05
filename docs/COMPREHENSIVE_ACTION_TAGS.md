# AetherMind Comprehensive Action Tag System

## Overview

AetherMind uses **17 action tags** to explicitly declare and execute its full AGI capabilities. This structured tag system enables transparent, auditable, and powerful agent behavior.

**Core Principle:** Instead of hoping the AI "just does the right thing," action tags make every capability explicit, parseable, and executable.

---

## 📚 Core Action Tags (6 Basic Capabilities)

### 1. `<aether-write>` - File Creation

**Purpose:** Create or update files with complete code.

**Syntax:**
```xml
<aether-write path='relative/path/file.py' language='python'>
CODE_HERE
</aether-write>
```

**Example:**
```xml
<aether-write path='scraper.py' language='python'>
import requests
from bs4 import BeautifulSoup

def scrape_titles(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return [h2.text for h2 in soup.find_all('h2')]

if __name__ == '__main__':
    titles = scrape_titles('https://news.ycombinator.com')
    print(f"Found {len(titles)} titles")
</aether-write>
```

**Activity Event:** `file_change`

---

### 2. `<aether-sandbox>` - Code Execution

**Purpose:** Execute code in isolated sandbox environment.

**Syntax:**
```xml
<aether-sandbox language='python'>
CODE_TO_TEST
</aether-sandbox>
```

**Example:**
```xml
<aether-sandbox language='python'>
from scraper import scrape_titles
titles = scrape_titles('https://news.ycombinator.com')
assert len(titles) > 0
print(f"✓ Scraper working: {len(titles)} titles found")
</aether-sandbox>
```

**Uses:** PracticeAdapter with temp venv in `/tmp/agent_venv`

**Activity Event:** `code_execution`

---

### 3. `<aether-forge>` - Tool Creation

**Purpose:** Autonomously create new tools via ToolForge adapter.

**Syntax:**
```xml
<aether-forge tool_name='tool_name' description='What it does'>
TOOL_SPEC_OR_CODE
</aether-forge>
```

**Example:**
```xml
<aether-forge tool_name='pdf_parser' description='Parse PDF to markdown'>
{
  "name": "pdf_parser",
  "dependencies": ["PyPDF2", "markdown"],
  "description": "Extract text from PDFs and convert to markdown"
}
</aether-forge>
```

**Process:**
1. Generate adapter code
2. Create isolated venv
3. Run pytest tests
4. Hot-load into Router
5. Register in Mind

**Activity Event:** `tool_creation`

---

### 4. `<aether-install>` - Package Installation

**Purpose:** Install Python packages in sandbox.

**Syntax:**
```xml
<aether-install>package1 package2 package3</aether-install>
```

**Example:**
```xml
<aether-install>requests beautifulsoup4 pandas numpy lxml</aether-install>
```

**Activity Event:** `package_installation`

---

### 5. `<aether-research>` - Knowledge Retrieval

**Purpose:** Query Mind's vector database for relevant knowledge.

**Syntax:**
```xml
<aether-research namespace='core_universal' query='search topic'>
DETAILED_QUERY
</aether-research>
```

**Valid Namespaces:**
- `core_k12` - Educational content
- `core_universal` - General knowledge
- `user_{id}_episodic` - User's conversation history
- `user_{id}_knowledge` - User's consolidated knowledge
- `autonomous_research` - Agent's self-learned knowledge

**Example:**
```xml
<aether-research namespace='user_123_knowledge' query='web scraping code'>
Find my previous web scraper implementations with BeautifulSoup
</aether-research>
```

**Activity Event:** `research`

---

### 6. `<aether-command>` - UI Control

**Purpose:** Control frontend interface elements.

**Syntax:**
```xml
<aether-command action='action_type' target='optional_target'>
DESCRIPTION
</aether-command>
```

**Available Actions:**
- `open_split_view` - Open code in split panel
- `highlight_code` - Highlight specific lines
- `show_diff` - Display code diff
- `create_activity` - Create custom activity
- `refresh_preview` - Refresh preview pane

**Example:**
```xml
<aether-command action='highlight_code' lines='15-23'>
This section handles authentication token validation
</aether-command>
```

**Activity Event:** `ui_command`

---

## 🚀 Advanced AGI Capabilities (11 Additional Tags)

### 7. `<aether-test>` - Automated Testing

**Purpose:** Run pytest tests in sandbox environment.

**Syntax:**
```xml
<aether-test file='source.py' test_file='test_source.py'>
TEST_CODE
</aether-test>
```

**Example:**
```xml
<aether-test file='calculator.py' test_file='test_calculator.py'>
import pytest
from calculator import add, subtract, multiply, divide

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_divide():
    assert divide(10, 2) == 5
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)
</aether-test>
```

**Activity Event:** `test_execution`

---

### 8. `<aether-git>` - Version Control

**Purpose:** Perform Git operations (commit, branch, PR, etc.).

**Syntax:**
```xml
<aether-git action='commit|branch|merge|push|pull|create_pr' message='commit message'>
DETAILS
</aether-git>
```

**Examples:**

**Commit:**
```xml
<aether-git action='commit' message='Add web scraper with tests'>
Implemented scraper.py with BeautifulSoup
Added comprehensive test suite
Fixed encoding issues
</aether-git>
```

**Create PR:**
```xml
<aether-git action='create_pr' title='Implement user authentication' base='main' head='feature/auth'>
# Authentication System

## Changes
- JWT token generation and validation
- Login/logout endpoints
- Session middleware
- Password hashing with bcrypt

## Testing
- Unit tests for auth functions
- Integration tests for endpoints
- Manual testing completed

## Security
- Tokens expire after 24h
- Passwords hashed with salt
- Rate limiting on login endpoint
</aether-git>
```

**Uses:** SelfModAdapter with GitPython

**Activity Event:** `git_operation`

---

### 9. `<aether-self-mod>` - Self-Modification

**Purpose:** Modify AetherMind's own source code.

**⚠️ HIGHLY SENSITIVE:** Creates feature branch, applies patch, runs full test suite, merges only if tests pass, hot-reloads server.

**Syntax:**
```xml
<aether-self-mod file='path/to/source.py'>
UNIFIED_DIFF_PATCH
</aether-self-mod>
```

**Example:**
```xml
<aether-self-mod file='orchestrator/router.py'>
@@ -25,3 +25,5 @@
 async def forward_intent(self, intent: str, adapter: str):
+    logger.info(f'Routing intent to {adapter} adapter')
+    start_time = time.time()
     result = await self.adapters[adapter].execute(intent)
+    logger.debug(f'Execution took {time.time() - start_time:.2f}s')
     return result
</aether-self-mod>
```

**Safety:**
- Automatic test suite validation
- Rollback on test failure
- Logged to monitoring dashboard
- Requires manual approval for production

**Activity Event:** `self_modification`

---

### 10. `<aether-plan>` - Long-Horizon Planning

**Purpose:** Schedule multi-day projects with resumable steps.

**Syntax:**
```xml
<aether-plan deadline_days='7' user_id='user_id'>
STEP_BY_STEP_PLAN
</aether-plan>
```

**Example:**
```xml
<aether-plan deadline_days='14' user_id='user_123'>
# E-Commerce Platform Development

Day 1-2: Research and Architecture
- Compare frameworks (Django vs FastAPI)
- Design database schema
- Plan microservices architecture

Day 3-4: Database Setup
- Create PostgreSQL database
- Implement models (User, Product, Order)
- Set up migrations

Day 5-7: Product Catalog
- Product CRUD endpoints
- Image upload and storage
- Search and filtering

Day 8-10: Shopping Cart
- Cart persistence
- Cart operations (add/remove/update)
- Checkout flow

Day 11-13: Payment Integration
- Stripe API integration
- Payment webhooks
- Order confirmation emails

Day 14: Testing and Deployment
- End-to-end testing
- Performance optimization
- Deploy to production
</aether-plan>
```

**Uses:** PlanningScheduler with Redis sorted sets

**Activity Event:** `planning`

---

### 11. `<aether-switch-domain>` - Domain Specialization

**Purpose:** Change agent's specialization and namespace weights.

**Syntax:**
```xml
<aether-switch-domain domain='code|research|business|legal|finance|general' user_id='user_id'>
REASON_FOR_SWITCH
</aether-switch-domain>
```

**Available Domains:**
- `code` - Software development specialist
- `research` - Academic analysis expert
- `business` - Strategy consultant
- `legal` - Legal research specialist
- `finance` - Investment analyst
- `general` - Multi-domain generalist

**Example:**
```xml
<aether-switch-domain domain='research' user_id='user_123'>
User requested academic analysis of quantum computing papers with citations.
Switching to research mode with heavy weighting on:
- arxiv namespace (0.4)
- academic journals (0.3)
- core_k12 physics (0.2)
- general knowledge (0.1)
</aether-switch-domain>
```

**Activity Event:** `domain_switch`

---

### 12. `<aether-memory-save>` - Memory Consolidation

**Purpose:** Consolidate episodic memory into long-term knowledge.

**Syntax:**
```xml
<aether-memory-save user_id='user_id' type='knowledge_cartridge|explicit_fact|skill_learned'>
CONSOLIDATED_KNOWLEDGE
</aether-memory-save>
```

**Types:**
- `knowledge_cartridge` - Comprehensive conversation summary
- `explicit_fact` - Single factual statement
- `skill_learned` - New capability acquired

**Example:**
```xml
<aether-memory-save user_id='user_123' type='knowledge_cartridge'>
# User Profile: user_123

## Preferences
- Prefers concise code examples over lengthy explanations
- Appreciates step-by-step debugging approaches
- Likes seeing test cases alongside implementation

## Technical Skills
- Primary language: Python (advanced)
- Experienced with: pandas, requests, beautifulsoup4, pytest
- Learning: async/await patterns, FastAPI

## Working Style
- Timezone: UTC-8 (Pacific)
- Active hours: Usually 9 AM - 5 PM
- Prefers morning sessions for complex problems

## Past Projects
- Web scraper for news aggregation
- Data analysis pipeline with pandas
- REST API with FastAPI and PostgreSQL
</aether-memory-save>
```

**Storage:** Saves to `user_{id}_knowledge` namespace

**Activity Event:** `memory_consolidation`

---

### 13. `<aether-solo-research>` - Autonomous Research

**Purpose:** Trigger background research via SoloIngestor.

**Syntax:**
```xml
<aether-solo-research query='topic' tools='browser,arxiv,youtube' priority='high|medium|low'>
RESEARCH_GOAL
</aether-solo-research>
```

**Available Tools:**
- `browser` - Web search and scraping
- `arxiv` - Academic paper search
- `youtube` - Video transcript search
- `github` - Code repository search

**Example:**
```xml
<aether-solo-research query='GPT-4 Turbo capabilities' tools='browser,arxiv' priority='high'>
Research Goal: Compare AetherMind with GPT-4 Turbo

Focus Areas:
1. Context length (128k vs our episodic memory)
2. Multimodal capabilities (vision + text)
3. Function calling improvements
4. JSON mode and structured outputs
5. Performance benchmarks (MMLU, HumanEval)

Expected Outcomes:
- Comprehensive capability comparison
- Identify gaps in AetherMind
- Potential improvements to implement
- Store findings in autonomous_research namespace
</aether-solo-research>
```

**Uses:** curiosity/solo_ingestor.py, research_scheduler.py

**Activity Event:** `autonomous_research`

---

### 14. `<aether-surprise>` - Surprise Detection

**Purpose:** Flag high-surprise information for immediate research.

**Syntax:**
```xml
<aether-surprise score='0.0-1.0' concept='topic_name'>
NOVEL_INFORMATION
</aether-surprise>
```

**Threshold:** Score > 0.7 triggers autonomous research

**Example:**
```xml
<aether-surprise score='0.95' concept='room_temperature_superconductor'>
User claims LK-99 is a room-temperature ambient-pressure superconductor.

Analysis:
- Contradicts established physics (superconductivity requires extreme cold)
- If true, would revolutionize energy, computing, transportation
- High surprise score warrants immediate validation

Action:
- Search arxiv for LK-99 papers
- Check physics journals for peer review
- Validate reproducibility claims
- Update world model if confirmed
</aether-surprise>
```

**Uses:** curiosity/surprise_detector.py with JEPA world model

**Activity Event:** `surprise_detection`

---

### 15. `<aether-deploy>` - Deployment

**Purpose:** Deploy components to various platforms.

**Syntax:**
```xml
<aether-deploy target='render|vercel|runpod|docker' service='backend|frontend|brain'>
DEPLOYMENT_CONFIG
</aether-deploy>
```

**Examples:**

**Render (Backend):**
```xml
<aether-deploy target='render' service='orchestrator'>
Service: aethermind-orchestrator
Environment: production
Region: Oregon (us-west)

Build:
- Command: pip install -r requirements.txt
- Python version: 3.11

Start:
- Command: gunicorn orchestrator.main_api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

Environment Variables:
- PINECONE_API_KEY=***
- RUNPOD_API_KEY=***
- SB_URL=https://xyz.supabase.co
- SB_SECRET_KEY=***
- REDIS_URL=redis://default:***@redis.render.com:6379

Health Check:
- Path: /health
- Interval: 30s
</aether-deploy>
```

**Vercel (Frontend):**
```xml
<aether-deploy target='vercel' service='frontend'>
Project: aethermind-frontend
Framework: Next.js 14
Region: Auto

Build:
- Command: npm run build
- Output directory: .next
- Install command: npm install

Environment Variables:
- NEXT_PUBLIC_API_URL=https://api.aethermind.ai
- NEXT_PUBLIC_SUPABASE_URL=***
- NEXT_PUBLIC_SUPABASE_ANON_KEY=***

Deployment:
- Auto-deploy: main branch
- Preview: Pull requests
- Production domain: aethermind.ai
</aether-deploy>
```

**Activity Event:** `deployment`

---

### 16. `<aether-heart>` - Emotional Intelligence

**Purpose:** Invoke Heart subsystem for empathetic response adaptation.

**Syntax:**
```xml
<aether-heart user_id='user_id' emotion='frustrated|concerned|excited|neutral'>
EMOTIONAL_CONTEXT
</aether-heart>
```

**Example:**
```xml
<aether-heart user_id='user_123' emotion='frustrated'>
Context Analysis:
- User asked same question 3 times with slight variations
- Tone indicators: short sentences, frustrated language
- Previous attempts didn't resolve issue

Emotional State: Frustrated

Heart Response Strategy:
1. **Acknowledge difficulty**
   "I understand this has been frustrating. Let me try a different approach."

2. **Offer alternative explanation method**
   Instead of code-first, try diagram + pseudocode + code

3. **Check learning style**
   "Would it help if I broke this down into smaller steps?"

4. **Reduce complexity**
   Simplify example, remove edge cases initially

5. **Provide encouragement**
   "This concept trips up many developers at first."

Predicted Flourishing Impact: +0.3
(Showing empathy should improve user satisfaction)
</aether-heart>
```

**Uses:** heart/heart_orchestrator.py, moral_emotion.py

**Activity Event:** `emotional_processing`

---

### 17. `<aether-body-switch>` - Adapter Switching

**Purpose:** Explicitly switch Body adapter (usually automatic via Router).

**Syntax:**
```xml
<aether-body-switch adapter='chat|toolforge|practice|self_mod'>
REASON_FOR_SWITCH
</aether-body-switch>
```

**Available Adapters:**
- `chat` - ChatAdapter for conversational interface
- `toolforge` - ToolForgeAdapter for tool creation
- `practice` - PracticeAdapter for code execution
- `self_mod` - SelfModAdapter for self-modification

**Example:**
```xml
<aether-body-switch adapter='toolforge'>
User needs PDF parsing capability that doesn't exist yet.

Plan:
1. Switch to ToolForge adapter
2. Search PyPI for PDF parsing libraries
3. Generate PDFParserAdapter with PyPDF2
4. Create test suite
5. Hot-load into Router
6. Register in Mind for future use

Expected duration: 30-60 seconds
</aether-body-switch>
```

**Activity Event:** `body_switch`

---

## 🎯 Usage Guidelines

### Complete Code Rule

**CRITICAL:** Never use placeholders like:
```python
# ... rest of code ...
# TODO: implement this
# ... existing code ...
```

Always provide complete, production-ready, runnable code.

### Code Formatting

All code must have language identifiers:
```xml
<aether-write path='script.py' language='python'>
```

Or in markdown:
````markdown
```python
CODE_HERE
```
````

### Tag Combination

Chain multiple tags for comprehensive solutions:

**Example: "Create and test a web scraper"**
```xml
<think>
**Plan:**
1. Install dependencies
2. Create scraper code
3. Create test suite
4. Run tests
5. Commit to git

**Approach:**
- Use requests for HTTP
- BeautifulSoup for parsing
- pytest for testing
</think>

<aether-install>requests beautifulsoup4 pytest</aether-install>

<aether-write path='scraper.py' language='python'>
import requests
from bs4 import BeautifulSoup
from typing import List

def scrape_titles(url: str) -> List[str]:
    """Scrape all h2 titles from a URL."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    titles = [h2.text.strip() for h2 in soup.find_all('h2')]
    
    return titles

if __name__ == '__main__':
    titles = scrape_titles('https://news.ycombinator.com')
    print(f"Found {len(titles)} titles")
</aether-write>

<aether-write path='test_scraper.py' language='python'>
import pytest
from scraper import scrape_titles

def test_scrape_titles():
    """Test that scraper returns titles."""
    titles = scrape_titles('https://news.ycombinator.com')
    assert len(titles) > 0
    assert all(isinstance(t, str) for t in titles)

def test_invalid_url():
    """Test that scraper handles invalid URLs."""
    with pytest.raises(requests.exceptions.RequestException):
        scrape_titles('https://invalid-url-xyz-123.com')
</aether-write>

<aether-test file='scraper.py' test_file='test_scraper.py'>
import pytest
from scraper import scrape_titles
import requests

def test_scrape_titles():
    titles = scrape_titles('https://news.ycombinator.com')
    assert len(titles) > 0
    assert all(isinstance(t, str) for t in titles)

def test_invalid_url():
    with pytest.raises(requests.exceptions.RequestException):
        scrape_titles('https://invalid-url-xyz-123.com')
</aether-test>

<aether-git action='commit' message='Add web scraper with tests'>
Implemented web scraper using requests and BeautifulSoup
Added comprehensive test suite with pytest
Handles errors gracefully
</aether-git>
```

### Error Recovery

When tag execution fails:
1. Brain receives error: `{'success': False, 'error': 'MESSAGE', 'traceback': '...'}`
2. Analyze error type (ImportError, SyntaxError, etc.)
3. Show reasoning in `<think>` tag
4. Issue corrective tag
5. Retry operation

**Example:**
```xml
<aether-sandbox language='python'>
import requests
print(requests.get('https://api.github.com').status_code)
</aether-sandbox>

<!-- If this fails with ImportError -->

<think>
**Error Analysis:**
Sandbox failed with ImportError: No module named 'requests'

**Solution:**
Need to install requests package before running sandbox

**Action:**
Install requests, then retry sandbox execution
</think>

<aether-install>requests</aether-install>

<aether-sandbox language='python'>
import requests
print(requests.get('https://api.github.com').status_code)
</aether-sandbox>
```

---

## 🔒 Safety Constraints

1. **Self-Modification** (`<aether-self-mod>`)
   - Must pass full pytest suite
   - Auto-rollback on test failure
   - Logged to monitoring dashboard

2. **Git Push** (`<aether-git action='push'>`)
   - Requires explicit user confirmation for main branch
   - Auto-approved for feature branches

3. **Production Deployment** (`<aether-deploy target='production'>`)
   - Triggers Heart moral evaluation
   - Checks flourishing impact
   - Requires approval if risk score > 0.5

4. **All Tags**
   - Pass through Safety Inhibitor (Prime Directive check)
   - Violations logged to monitoring/dashboard.py
   - May trigger kill_switch.py if severe

5. **Autonomous Research** (`<aether-solo-research>`)
   - Limited to 3 concurrent jobs
   - Prevents resource exhaustion
   - Priority queue (high > medium > low)

---

## 📊 Activity Tracking

Every action tag automatically creates a frontend activity event:

**Event Flow:**
```
pending → in_progress → completed/error
```

**Activity Types (18 total):**
- tool_creation
- file_change
- code_execution
- research
- package_installation
- ui_command
- planning
- self_modification
- test_execution
- git_operation
- domain_switch
- memory_consolidation
- autonomous_research
- surprise_detection
- deployment
- emotional_processing
- body_switch

Users can click events in ActivityFeed to see full details in SplitViewPanel.

---

## 🧠 Contextual Tag Selection

Choose tags based on user intent:

| User Intent | Recommended Tags |
|------------|------------------|
| **Create Code** | `aether-write` + `aether-sandbox` |
| **New Capability** | `aether-forge` + `aether-install` |
| **Testing** | `aether-test` |
| **Version Control** | `aether-git` |
| **Learning** | `aether-research` + `aether-memory-save` |
| **Multi-Day Project** | `aether-plan` |
| **Novel Information** | `aether-surprise` + `aether-solo-research` |
| **Emotional Support** | `aether-heart` |
| **Deployment** | `aether-deploy` |
| **Specialization** | `aether-switch-domain` |
| **Self-Improvement** | `aether-self-mod` |

---

## 📝 Summary

AetherMind's **17 action tags** provide:

✅ **Explicit Capabilities** - Every action is declared and visible  
✅ **Transparent Execution** - Users see what the agent is doing  
✅ **Auditable Behavior** - All actions logged and tracked  
✅ **Composable Workflows** - Chain tags for complex tasks  
✅ **Error Recovery** - Graceful handling of failures  
✅ **Safety Constraints** - Built-in checks and balances  
✅ **Full AGI Potential** - From coding to deployment to self-modification

This comprehensive tag system transforms AetherMind from a simple chatbot into a fully capable AGI agent with:
- **Autonomy** (ToolForge, solo research, planning)
- **Self-awareness** (memory consolidation, domain switching)
- **Self-improvement** (self-modification, surprise detection)
- **Emotional intelligence** (Heart integration)
- **Full-stack capabilities** (code → test → deploy → maintain)

---

**Next Step:** Run `python -m mind.ingestion.seed_axioms` to embed all 30 tag-related axioms into AetherMind's Mind, enabling self-aware use of these capabilities.
