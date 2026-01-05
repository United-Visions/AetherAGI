# AetherMind Action Tag System

## Overview

Similar to Dyad's `<dyad-write>` tags, AetherMind now uses **structured action tags** to make the agent's actions explicit, trackable, and executable. This system provides:

- **Clear Intent**: Brain explicitly declares what actions to take
- **Automatic Execution**: Tags are parsed and executed automatically
- **Frontend Visibility**: All actions tracked in Activity Stream
- **Code Formatting**: All code in proper markdown blocks or action tags

## Architecture

```
User Request
    ↓
Active Inference Loop
    ↓
Brain generates response with ACTION TAGS
    ↓
ActionParser extracts tags
    ↓
ActionExecutor runs each action
    ↓
Activity Events sent to frontend
    ↓
User sees real-time updates
```

## Available Action Tags

### 1. File Creation/Modification

```xml
<aether-write path="scraper.py" language="python" description="Web scraper">
import requests
from bs4 import BeautifulSoup

def scrape(url):
    response = requests.get(url)
    return BeautifulSoup(response.content, 'html.parser')
</aether-write>
```

**When to use:** Creating or updating files in the project

### 2. Sandbox Execution

```xml
<aether-sandbox language="python" test="true">
from scraper import scrape

result = scrape("https://example.com")
assert result is not None
print("✓ Test passed!")
</aether-sandbox>
```

**When to use:** Testing code in isolated environment

### 3. Tool Creation (ToolForge)

```xml
<aether-forge tool="weather_scraper" action="generate">
{
  "name": "weather_scraper",
  "description": "Scrapes weather data",
  "dependencies": ["requests", "beautifulsoup4"],
  "url": "https://api.weather.com"
}
</aether-forge>
```

**When to use:** Creating new capabilities dynamically

### 4. Package Installation

```xml
<aether-install packages="requests beautifulsoup4 pandas"></aether-install>
```

**When to use:** Installing external dependencies

### 5. Research/Memory Query

```xml
<aether-research query="Python async best practices" namespace="core_knowledge"></aether-research>
```

**When to use:** Retrieving information from Mind

### 6. System Commands

```xml
<aether-command type="refresh"></aether-command>
<aether-command type="restart"></aether-command>
```

**When to use:** Suggesting UI actions to user

## Code Formatting Rules

### Markdown Code Blocks (Preferred for Display)

✅ **CORRECT:**
\`\`\`python
def hello():
    print("Hello, World!")
\`\`\`

✅ **CORRECT:**
\`\`\`javascript
const greet = () => {
    console.log("Hello, World!");
};
\`\`\`

❌ **WRONG:**
```
Plain code without language tag
```

### Action Tags (For Execution)

When code needs to be created as a file or executed:

\`\`\`xml
<aether-write path="hello.py" language="python" description="Hello world script">
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
</aether-write>
\`\`\`

## System Prompt Integration

### Brain's System Prompt Now Includes:

1. **Thinking Process** (`<think></think>` tags)
2. **Action Tag Definitions** (what each tag does)
3. **Code Formatting Rules** (mandatory language tags)
4. **Domain-Specific Behavior** (code, research, business, etc.)
5. **Implementation Guidelines** (complete code, no placeholders)

### Example Brain Response:

```
User: "Create a web scraper for Hacker News"

<think>
• **Identify requirements**
  - Scrape article titles and URLs
  - Handle network errors
  - Parse HTML structure

• **Plan implementation**
  - Use requests + BeautifulSoup
  - Create reusable scraper class
  - Add error handling and logging

• **Test approach**
  - Create sandbox test
  - Validate HTML parsing
  - Check output format
</think>

I'll create a complete Hacker News scraper with proper error handling.

<aether-install packages="requests beautifulsoup4"></aether-install>

<aether-write path="hn_scraper.py" language="python" description="Hacker News scraper">
import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HNScraper:
    \"\"\"Scrapes articles from Hacker News.\"\"\"
    
    def __init__(self):
        self.base_url = "https://news.ycombinator.com"
        self.session = requests.Session()
    
    def scrape_front_page(self):
        \"\"\"Get front page articles.\"\"\"
        try:
            response = self.session.get(self.base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            for row in soup.select('.athing'):
                title_elem = row.select_one('.titleline > a')
                if title_elem:
                    articles.append({
                        'title': title_elem.text,
                        'url': title_elem['href']
                    })
            
            logger.info(f"Scraped {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return []

if __name__ == "__main__":
    scraper = HNScraper()
    articles = scraper.scrape_front_page()
    for article in articles[:5]:
        print(f"{article['title']}: {article['url']}")
</aether-write>

<aether-sandbox language="python" test="true">
from hn_scraper import HNScraper

scraper = HNScraper()
articles = scraper.scrape_front_page()

assert len(articles) > 0, "Should find articles"
assert 'title' in articles[0], "Articles need titles"
assert 'url' in articles[0], "Articles need URLs"

print(f"✓ Scraper works! Found {len(articles)} articles")
</aether-sandbox>

<aether-chat-summary>Created HN scraper</aether-chat-summary>

I've created a complete Hacker News scraper with error handling and testing.
```

## Implementation Details

### Files Added/Modified:

1. **`brain/system_prompts.py`** - Structured prompts with action tag definitions
2. **`orchestrator/action_parser.py`** - Parses and executes action tags
3. **`orchestrator/active_inference.py`** - Integrated parsing/execution
4. **`orchestrator/session_manager.py`** - Provides structured prompts
5. **`brain/logic_engine.py`** - Uses structured prompts

### Action Parser Flow:

```python
# 1. Parse tags from response
action_tags, cleaned_response = action_parser.parse(brain_response)

# 2. Convert to activity events
for tag in action_tags:
    event = tag.to_activity_event(user_id)
    activity_events.append(event)

# 3. Execute each action
for tag in action_tags:
    result = await action_executor.execute(tag, user_id)
    # Update activity status based on result

# 4. Extract code blocks (for display)
code_blocks = action_parser.extract_code_blocks(brain_response)

# 5. Return cleaned response + activities
return cleaned_response, activity_events
```

### Activity Event Schema:

```javascript
{
  id: "aether-write_2026-01-05T12:34:56",
  type: "file_change",  // or code_execution, tool_creation, etc.
  status: "completed",  // or in_progress, error
  title: "Creating scraper.py",
  details: "Web scraper with error handling",
  timestamp: "2026-01-05T12:34:56.789Z",
  data: {
    files: ["scraper.py"],
    code: "import requests\n...",
    language: "python"
  }
}
```

## Benefits

### 1. **Explicit Actions**
- No more guessing what the agent is doing
- Clear distinction between explanation and execution
- Trackable action history

### 2. **Automatic Execution**
- Tags parsed and executed automatically
- Results captured and reported
- Errors handled gracefully

### 3. **Frontend Visibility**
- Activity Stream shows all actions in real-time
- Click cards to see full code/details
- Progress tracking for multi-step operations

### 4. **Code Quality**
- All code properly formatted with language tags
- Syntax highlighting in UI
- Easy copy/paste for users

### 5. **Debugging**
- Action tags logged for audit trail
- Execution results captured
- Easy to replay or modify actions

## Usage Examples

### Example 1: Simple Script

**User:** "Create a hello world script"

**Agent Response:**
```
<aether-write path="hello.py" language="python" description="Hello world">
print("Hello, World!")
</aether-write>

Created a simple hello world script.
```

**Activity Stream:**
- 📄 Creating hello.py (completed)

### Example 2: Full Web App

**User:** "Build a Flask todo API"

**Agent Response:**
```
<think>
• Need REST endpoints (GET, POST, PUT, DELETE)
• Use in-memory storage for simplicity
• Add CORS for frontend access
</think>

<aether-install packages="flask flask-cors"></aether-install>

<aether-write path="app.py" language="python" description="Flask todo API">
[... full API code ...]
</aether-write>

<aether-sandbox language="python" test="true">
[... test code ...]
</aether-sandbox>

<aether-chat-summary>Built Flask todo API</aether-chat-summary>
```

**Activity Stream:**
- 📦 Installing 2 packages (completed)
- 📄 Creating app.py (completed)
- ▶️ Executing code in sandbox (completed)

### Example 3: Tool Creation

**User:** "I need a tool to scrape product prices"

**Agent Response:**
```
<aether-forge tool="price_scraper" action="generate">
{
  "name": "price_scraper",
  "description": "Scrapes product prices from e-commerce sites",
  "dependencies": ["requests", "beautifulsoup4", "lxml"]
}
</aether-forge>

<aether-write path="price_scraper.py" language="python" description="Price scraper tool">
[... scraper implementation ...]
</aether-write>

Created a price scraper tool. You can now use it with: price_scraper.scrape(url)
```

**Activity Stream:**
- 🔧 Forging price_scraper (generate) (completed)
- 📄 Creating price_scraper.py (completed)

## Testing

To test the action tag system:

1. **Start backend:**
   ```bash
   cd orchestrator
   python main_api.py
   ```

2. **Send test request:**
   ```python
   # User message: "Create a web scraper"
   # Brain will respond with action tags
   # Tags will be parsed and executed
   # Activity events sent to frontend
   ```

3. **Check Activity Stream:**
   - Should show: Installing packages, Creating files, Executing code
   - Click cards to see full code with syntax highlighting

## Future Enhancements

1. **Streaming Tags** - Stream action tags as they're generated
2. **Interactive Approval** - Ask user before executing actions
3. **Rollback** - Undo actions if something goes wrong
4. **Action History** - Store all actions in Mind for replay
5. **Multi-Agent Tags** - Coordinate between multiple agents
6. **Visual Builder** - Drag-drop interface for action tags

## Related Documentation

- [AGI Capabilities](AGI_CAPABILITIES.md) - Full list of agent capabilities
- [Sandbox Lifecycle](SANDBOX_LIFECYCLE.md) - How sandboxes are created/managed
- [Activity Stream Enhancement](ACTIVITY_STREAM_ENHANCEMENT.md) - Frontend tracking
