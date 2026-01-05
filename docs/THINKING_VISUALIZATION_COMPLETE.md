# AetherMind Thinking Visualization - Complete Integration

## Overview

AetherMind now has **complete thinking visualization** integrated throughout the entire system - from Brain generation → Backend parsing → Frontend display. Users can see the agent's reasoning process in real-time before the final response is displayed.

---

## 🧠 How It Works

### 1. **Brain Generates Thinking Tags**

The Brain is prompted (via `brain/system_prompts.py`) to use `<think>` tags to show its reasoning:

```xml
<think>
**Analysis**: User wants a web scraper
**Approach**: 
1. Use requests for HTTP
2. BeautifulSoup for parsing
3. Create sandbox for testing
**Implementation**: 
- Write scraper.py
- Install dependencies
- Test with sandbox
</think>
```

**Axiom Reference:**
```python
{"subject": "AetherMind", "topic": "ThinkingTags", "content": "Always use <think>REASONING</think> to show planning process. Appears in frontend Thought Bubble. Example: <think>**Analysis**: User wants snake game\n**Approach**: 1) pygame for rendering, 2) game loop with event handling, 3) collision detection for food/walls\n**Implementation**: Create main.py with Game class, separate Snake and Food classes</think> Thinking tags visible but don't trigger execution."}
```

---

### 2. **Backend Extracts Thinking**

**File:** `orchestrator/action_parser.py`

The `ActionParser` has a dedicated `parse_thinking()` method that:
- Extracts all `<think>` tags from Brain response
- Splits multi-line thinking into bullet points
- Returns cleaned response (without thinking tags)

```python
def parse_thinking(self, brain_response: str) -> Tuple[List[str], str]:
    """
    Extract <think> tags and return thinking steps + cleaned response.
    
    Returns:
        (thinking_steps, cleaned_response)
    """
    thinking_steps = []
    cleaned_response = brain_response
    
    # Extract all <think> tags
    think_pattern = r'<think>(.*?)</think>'
    matches = re.finditer(think_pattern, brain_response, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        thinking_content = match.group(1).strip()
        
        # Split into bullet points or lines
        if '\n' in thinking_content:
            # Multi-line thinking
            lines = [line.strip() for line in thinking_content.split('\n') if line.strip()]
            thinking_steps.extend(lines)
        else:
            # Single line
            thinking_steps.append(thinking_content)
        
        # Remove from cleaned response
        cleaned_response = cleaned_response.replace(match.group(0), "")
    
    # Clean up extra whitespace
    cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response).strip()
    
    logger.info(f"Extracted {len(thinking_steps)} thinking steps")
    return thinking_steps, cleaned_response
```

**TAG_PATTERNS includes:**
```python
"think": r'<think>(.*?)</think>',  # Thinking process visualization
```

---

### 3. **Active Inference Loop Processes Thinking**

**File:** `orchestrator/active_inference.py`

The Active Inference Loop extracts thinking before action tags:

```python
# 5. EMBELLISH: Heart adapts the response based on emotion and morals
embellished_response = self.heart.embellish_response(brain_response, emotion_vector, predicted_flourishing)

# 6. EXTRACT THINKING: Parse <think> tags to show agent's reasoning process
thinking_steps, response_without_thinking = self.action_parser.parse_thinking(embellished_response)

# 7. PARSE ACTION TAGS: Extract structured actions from Brain response
action_tags, cleaned_response = self.action_parser.parse(response_without_thinking)

# ... execute actions ...

# 11. PREPARE FOR LEARNING: Cache the data needed for the feedback loop
self.last_trace_data[emotion_vector["message_id"]] = {
    "state_vector": state_vec,
    "action_text": final_output,
    "predicted_flourishing": predicted_flourishing,
    "user_id": user_id,
    "thinking_steps": thinking_steps  # Store reasoning process
}

# 13. UPDATE AGENT STATE WITH THINKING
agent_state["thinking_steps"] = thinking_steps
agent_state["action_count"] = len(action_tags)
agent_state["activity_events"] = self.activity_events
```

**Flow:**
1. Brain generates response with `<think>` tags
2. Heart embellishes
3. **Thinking extracted first** (preserves reasoning)
4. Action tags parsed and executed
5. Thinking stored in `agent_state`

---

### 4. **API Sends Thinking to Frontend**

**File:** `orchestrator/main_api.py`

The `/v1/chat/completions` endpoint extracts and includes thinking:

```python
# Run the DCLA Logic Cycle
response_text, message_id, emotion_vector, agent_state = await AETHER.run_cycle(user_id, last_message)

# Get activity events (tool creation, file changes, code execution)
activity_events = AETHER.get_activity_events()

# Extract thinking steps for frontend display
thinking_steps = agent_state.get("thinking_steps", []) if agent_state else []

# Return in a standardized format
return {
    "id": message_id,
    "object": "chat.completion",
    "model": request.model,
    "choices": [{
        "message": {
            "role": "assistant",
            "content": response_text  # CLEAN response without <think> tags
        },
        "finish_reason": "stop"
    }],
    "usage": {"total_tokens": len(response_text) // 4},
    "metadata": {
        "user_emotion": emotion_vector,
        "agent_state": agent_state,
        "activity_events": activity_events,
        "reasoning_steps": thinking_steps  # <think> tag content for visualization
    }
}
```

**Key Points:**
- `choices[0].message.content` = clean response (no thinking tags)
- `metadata.reasoning_steps` = extracted thinking steps
- `metadata.activity_events` = action tag executions

---

### 5. **Frontend Displays Thinking**

**File:** `frontend_flask/static/js/router.js`

The router processes thinking and displays it:

```javascript
// Activity Feed: Log thinking process
const thinkingActivityId = activityFeed.addActivity({
    id: `thinking_${Date.now()}`,
    type: 'thinking',
    status: 'in_progress',
    title: 'Processing your request',
    details: 'Running active inference loop...',
    timestamp: new Date().toISOString(),
    data: {
        reasoning: [
            '1. Sensing: Parsing user intent',
            '2. Retrieving: Searching mind for relevant knowledge',
            '3. Reasoning: Applying logic and domain expertise',
            '4. Embellishing: Adding emotional intelligence',
            '5. Acting: Generating response',
            '6. Learning: Storing to episodic memory'
        ]
    }
});

// Brain Visualizer: Start thinking animation
brainViz.startThinking();

// ... API call ...

// Update thinking activity with actual reasoning steps
activityFeed.updateActivity(thinkingActivityId, {
    status: 'completed',
    details: 'Response generated successfully',
    completed_at: new Date().toISOString(),
    data: {
        reasoning: metadata.reasoning_steps || [],  // REAL thinking from backend
        confidence: metadata.agent_state?.confidence || 0.85,
        surprise: metadata.agent_state?.surprise_score || 0
    }
});
```

**Components Involved:**
1. **ActivityFeed** - Shows thinking as an activity card
2. **BrainVisualizer** - Animates the 6-stage inference loop
3. **ThinkingVisualizer** - Creates detailed thinking cards
4. **SplitViewPanel** - Displays reasoning when clicked

---

## 📊 Complete Data Flow

```
User Input
    ↓
[1] Active Inference Loop
    ↓
[2] Brain.generate_thought()
    ↓
    "Let me help with that.
    <think>
    **Analysis**: User wants web scraper
    **Approach**: requests + BeautifulSoup
    **Steps**: 1) Install deps, 2) Write code, 3) Test
    </think>
    
    <aether-install>requests beautifulsoup4</aether-install>
    <aether-write path='scraper.py' language='python'>
    import requests
    from bs4 import BeautifulSoup
    ...
    </aether-write>"
    ↓
[3] ActionParser.parse_thinking()
    → thinking_steps = [
        "**Analysis**: User wants web scraper",
        "**Approach**: requests + BeautifulSoup",
        "**Steps**: 1) Install deps, 2) Write code, 3) Test"
      ]
    → cleaned_response = "Let me help with that.\n<aether-install>..."
    ↓
[4] ActionParser.parse()
    → action_tags = [
        ActionTag(type='aether-install', content='requests beautifulsoup4'),
        ActionTag(type='aether-write', content='import requests...', attributes={'path': 'scraper.py'})
      ]
    → cleaned_response = "Let me help with that."
    ↓
[5] ActionExecutor.execute()
    → Install packages
    → Create scraper.py file
    → Generate activity events
    ↓
[6] agent_state update
    agent_state = {
        "thinking_steps": [...],  # Extracted thinking
        "action_count": 2,
        "activity_events": [...],  # Action executions
        "surprise_score": 0.3,
        ...
    }
    ↓
[7] API Response
    {
        "choices": [{
            "message": {
                "content": "Let me help with that."  # CLEAN
            }
        }],
        "metadata": {
            "reasoning_steps": [...],  # THINKING
            "activity_events": [...]   # ACTIONS
        }
    }
    ↓
[8] Frontend Display
    ┌─────────────────────────────────┐
    │ 🧠 Thinking (completed)         │
    │ Response generated successfully │
    │ ├─ **Analysis**: User wants...  │
    │ ├─ **Approach**: requests + BS  │
    │ └─ **Steps**: 1) Install...     │
    └─────────────────────────────────┘
    ┌─────────────────────────────────┐
    │ 📦 Installing 2 packages        │
    │ Status: ✓ Completed             │
    └─────────────────────────────────┘
    ┌─────────────────────────────────┐
    │ 📝 Creating scraper.py          │
    │ [View Code]                     │
    └─────────────────────────────────┘
    ┌─────────────────────────────────┐
    │ 💬 Agent Response               │
    │ Let me help with that.          │
    └─────────────────────────────────┘
```

---

## 🎨 Frontend Components

### ActivityFeed.js

Shows thinking as an activity with icon:

```javascript
const iconMap = {
    'thinking': 'fas fa-brain',
    'tool_creation': 'fas fa-tools',
    'file_change': 'fas fa-file-code',
    // ... 18 activity types total
};
```

**Thinking Activity Structure:**
```javascript
{
    id: 'thinking_1704452340000',
    type: 'thinking',
    status: 'completed',
    title: 'Processing your request',
    details: 'Response generated successfully',
    timestamp: '2026-01-05T10:30:40.000Z',
    data: {
        reasoning: [
            '**Analysis**: User wants web scraper',
            '**Approach**: Use requests + BeautifulSoup',
            '**Steps**: 1) Install, 2) Write, 3) Test'
        ],
        confidence: 0.92,
        surprise: 0.15
    }
}
```

### BrainVisualizer.js

Animates the 6-stage Active Inference Loop:
1. **Sense** (green) - Parse intent
2. **Retrieve** (blue) - Search Mind
3. **Reason** (purple) - Apply logic
4. **Embellish** (pink) - Add emotion
5. **Act** (orange) - Generate response
6. **Learn** (cyan) - Store memory

Shows metrics:
- **Surprise Score** (0.00-1.00, red if > 0.5)
- **Confidence** (0-100%, red if < 50%)
- **Processing Time** (milliseconds)

### SplitViewPanel.js

When user clicks thinking activity, displays:

```html
<div class="split-view-content">
    <div class="activity-detail">
        <h2>🧠 Thinking Process</h2>
        <div class="activity-section">
            <h3><i class="fas fa-brain"></i> Agent Reasoning</h3>
            <div class="reasoning-steps">
                <div class="reasoning-step">
                    <span class="step-number">1</span>
                    <span class="step-text">**Analysis**: User wants web scraper</span>
                </div>
                <div class="reasoning-step">
                    <span class="step-number">2</span>
                    <span class="step-text">**Approach**: Use requests + BeautifulSoup</span>
                </div>
                <!-- ... more steps ... -->
            </div>
        </div>
    </div>
</div>
```

---

## 🔧 Configuration

### System Prompts (brain/system_prompts.py)

Includes thinking instructions:

```python
THINKING_PROMPT = """
## Thinking Process

Before responding, ALWAYS show your reasoning using <think> tags.

**Format:**
<think>
**Analysis**: [What is the user really asking for?]
**Approach**: [High-level strategy to solve this]
**Steps**: [Specific actions you'll take]
**Considerations**: [Edge cases, alternatives, trade-offs]
</think>

This makes your reasoning transparent and helps users understand your logic.
"""

AETHER_SYSTEM_PREFIX = f"""
{THINKING_PROMPT}

You are AetherMind, a Digital Organism with...
"""
```

### Seed Axioms (mind/ingestion/seed_axioms.py)

Axiom teaches the Brain to use thinking tags:

```python
{"subject": "AetherMind", "topic": "ThinkingTags", "content": "Always use <think>REASONING</think> to show planning process. Appears in frontend Thought Bubble. Example: <think>**Analysis**: User wants snake game\n**Approach**: 1) pygame for rendering, 2) game loop with event handling, 3) collision detection for food/walls\n**Implementation**: Create main.py with Game class, separate Snake and Food classes</think> Thinking tags visible but don't trigger execution."}
```

---

## ✅ Complete Integration Checklist

- [x] **Brain Generation** - System prompts include `<think>` tag instructions
- [x] **Knowledge Base** - ThinkingTags axiom seeded in core_universal
- [x] **Parsing** - `ActionParser.parse_thinking()` extracts thinking
- [x] **Tag Patterns** - `"think": r'<think>(.*?)</think>'` in TAG_PATTERNS
- [x] **Active Inference** - Thinking extracted before action tags
- [x] **Agent State** - `thinking_steps` stored in agent_state
- [x] **API Response** - `reasoning_steps` included in metadata
- [x] **Frontend Components** - ActivityFeed, BrainVisualizer, SplitViewPanel
- [x] **Router Integration** - Processes `reasoning_steps` from metadata
- [x] **Real-time Display** - Thinking shown BEFORE final response
- [x] **Activity Types** - "thinking" type with brain icon
- [x] **CSS Styling** - `.reasoning-steps`, `.reasoning-step` classes

---

## 🎯 Usage Examples

### Example 1: Simple Request

**User:** "Create a hello world app"

**Brain Response:**
```xml
<think>
**Analysis**: User wants basic application
**Approach**: Keep it simple with Flask
**Steps**: 1) Create app.py, 2) Add route, 3) Provide run instructions
</think>

<aether-write path='app.py' language='python'>
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
</aether-write>

I've created a simple Flask hello world app. Run with `python app.py` and visit http://localhost:5000
```

**Frontend Shows:**
1. 🧠 **Thinking** - 3 reasoning steps displayed
2. 📝 **Creating app.py** - File change activity
3. 💬 **Agent Response** - Clean message

---

### Example 2: Complex Request

**User:** "Build a complete e-commerce API with authentication"

**Brain Response:**
```xml
<think>
**Analysis**: Complex multi-component system required
**Scope**: 
- Database models (User, Product, Order)
- JWT authentication
- CRUD endpoints for products
- Shopping cart logic
- Payment integration placeholder

**Approach**: 
1. FastAPI for modern async API
2. SQLAlchemy for ORM
3. Pydantic for validation
4. JWT tokens for auth

**Architecture**:
- models.py (database schemas)
- auth.py (authentication logic)
- main.py (API routes)
- database.py (connection setup)

**Steps**:
1. Install dependencies
2. Create database models
3. Implement auth system
4. Build product endpoints
5. Add cart functionality
6. Create main API file
7. Provide deployment guide

**Considerations**:
- Input validation critical for security
- Password hashing required
- Rate limiting needed for production
- Consider adding tests
</think>

<aether-plan deadline_days='3' user_id='user_123'>
Day 1: Database and Models
- Set up PostgreSQL
- Create models (User, Product, Order, CartItem)
- Implement migrations

Day 2: Authentication
- JWT token generation
- Login/logout endpoints
- Password hashing with bcrypt
- Middleware for protected routes

Day 3: API Endpoints
- Product CRUD operations
- Cart management
- Order processing
- Testing and documentation
</aether-plan>

<aether-install>fastapi uvicorn sqlalchemy pydantic python-jose passlib bcrypt</aether-install>

<aether-write path='models.py' language='python'>
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    orders = relationship('Order', back_populates='user')

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)
    
class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    total = Column(Float)
    status = Column(String, default='pending')
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship('User', back_populates='orders')
</aether-write>

<aether-write path='auth.py' language='python'>
# ... complete authentication code ...
</aether-write>

<aether-git action='commit' message='Initial e-commerce API setup'>
Created foundational e-commerce API with:
- Database models for User, Product, Order
- JWT authentication system
- Password hashing with bcrypt
- SQLAlchemy ORM setup
</aether-git>

I've created a complete e-commerce API foundation with authentication. The system includes...
```

**Frontend Shows:**
1. 🧠 **Thinking** - Comprehensive 15+ step reasoning process
2. 📅 **Planning 3-day project** - Multi-day plan activity
3. 📦 **Installing 6 packages** - Dependency installation
4. 📝 **Creating models.py** - Database models file
5. 📝 **Creating auth.py** - Authentication system
6. 🔄 **Git commit** - Version control activity
7. 💬 **Agent Response** - Clean summary

---

## 🚀 Benefits

### For Users

1. **Transparency** - See exactly how AetherMind is thinking
2. **Trust** - Understand the reasoning behind responses
3. **Learning** - Learn problem-solving approaches from the agent
4. **Debugging** - Identify where thinking might be off-track
5. **Progress Tracking** - See multi-step workflows unfold

### For Developers

1. **Debugging** - See Brain's reasoning without logs
2. **Prompt Engineering** - Improve system prompts based on thinking quality
3. **Performance Metrics** - Track reasoning complexity and time
4. **Error Analysis** - Understand why certain responses fail
5. **Capability Discovery** - See when agent uses advanced features

---

## 📝 Summary

AetherMind's thinking visualization is now **fully integrated** across the entire stack:

✅ **Generation** - Brain prompted to use `<think>` tags via system prompts + axioms  
✅ **Parsing** - ActionParser extracts thinking separately from action tags  
✅ **Processing** - Active Inference Loop includes thinking in agent_state  
✅ **Transmission** - API sends reasoning_steps in metadata  
✅ **Display** - Frontend shows thinking in ActivityFeed + BrainVisualizer  
✅ **Interaction** - Users can click thinking activities to see details  

**Result:** Users see the agent's **complete thought process** streaming in real-time as it:
- Analyzes their request
- Plans its approach
- Executes actions
- Generates the response

This creates a **transparent, trustworthy, and educational** AI experience where users understand not just *what* the agent is doing, but **why** and **how** it's doing it.

---

**Next Step:** Run `python -m mind.ingestion.seed_axioms` to embed the ThinkingTags axiom, enabling the Brain to learn and use thinking tags correctly! 🚀
