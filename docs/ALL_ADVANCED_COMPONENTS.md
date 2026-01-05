# AetherMind Advanced Frontend Components - Complete Guide

**Status:** ✅ ALL 6 COMPONENTS IMPLEMENTED

This document provides a complete reference for all advanced frontend components built for AetherMind's AGI-level interface.

---

## Component #1: Streaming Accordion (Replaces "Thinking Process")

**File:** `static/js/components/StreamingAccordion.js`

Displays AetherMind's cognitive processes in an expandable, animated accordion layout.

### Supported Content Types

1. **Text**: Plain text streaming
2. **Code**: Syntax-highlighted code blocks
3. **Metrics**: Progress bars and percentage displays
4. **Graphs**: Emotion/confidence/surprise line graphs
5. **Lists**: Ordered and unordered lists
6. **Diffs**: Git-style before/after comparisons
7. **Emotions**: Emotion labels with animated intensity bars

### Features Displayed

- **Feature 15**: Differential Learning (before/after knowledge)
- **Feature 13**: Surprise Meter (prediction error tracking)
- **Feature 11**: Multi-Modal Reasoning (text, code, visual)
- **Feature 5**: Cognitive Process (step-by-step reasoning)

### Usage

```javascript
import { StreamingAccordion } from './components/StreamingAccordion.js';

const accordion = new StreamingAccordion('accordion-container');

// Create panel
accordion.createPanel('memory-query', 'Memory Query', 'fas fa-database', 'Searching...');

// Stream text
accordion.streamToPanel('memory-query', 'Found 3 relevant memories...');

// Append emotion graph
accordion.appendEmotion('memory-query', 'Confidence', 0.92, '#10b981');

// Append code
accordion.appendCode('memory-query', 'def example():\n    return "code"', 'python');

// Close panel
accordion.closePanel('memory-query', 'success', 'Query complete');
```

---

## Component #2: Live Activity Feed (User's Favorite!)

**File:** `static/js/components/LiveActivityFeed.js`

Horizontal scrolling real-time activity tracker showing what AetherMind is doing.

### Activity Types

1. **tool_create**: Creating new tools
2. **memory_update**: Updating knowledge base
3. **research**: Conducting research
4. **learning**: Active learning events
5. **self_modify**: Self-modification events
6. **knowledge_query**: Querying knowledge base
7. **safety_check**: Safety validation
8. **tool_execute**: Executing tools
9. **file_process**: Processing files
10. **conversation**: Conversation events

### Features

- Sliding card animations
- Status updates (pending → running → complete)
- Modal detailed views with code/logs/files
- Timeline history
- Activity grouping

### Usage

```javascript
import { LiveActivityFeed } from './components/LiveActivityFeed.js';

const feed = new LiveActivityFeed('activity-feed-container');

// Add activity
const activityId = feed.addActivity('tool_create', 'Building calculator tool', {
    tool_name: 'calculator',
    language: 'python'
});

// Update status
feed.updateActivity(activityId, 'running', 'Generating code...');
feed.updateActivity(activityId, 'complete', 'Tool created successfully');

// Show detailed view
feed.showActivityDetails(activityId);
```

---

## Component #3: Knowledge Graph (Advanced Feature #9)

**File:** `static/js/components/KnowledgeGraph.js`

Interactive force-directed graph visualizing AetherMind's knowledge connections.

### Features

- Canvas-based rendering with physics simulation
- Color-coded nodes:
  - Green: User memories
  - Blue: Core knowledge
  - Yellow glow: Recently accessed
- Drag-and-drop interactions
- Node inspection sidebar
- Auto-layout with force-directed physics

### Usage

```javascript
import { KnowledgeGraph } from './components/KnowledgeGraph.js';

const graph = new KnowledgeGraph('knowledge-graph-container');

// Add nodes
graph.addNode('node1', 'Python Loops', 'user', 0.95);
graph.addNode('node2', 'For Statement', 'core', 1.0);

// Add edge
graph.addEdge('node1', 'node2', 0.8);

// Update from memory retrieval
graph.updateFromMemory([
    { id: 'mem1', content: 'User preference', namespace: 'user', score: 0.9 },
    { id: 'mem2', content: 'Python syntax', namespace: 'core', score: 0.95 }
]);
```

---

## Component #4: Collaborative Sandbox Workspace (Advanced Feature #10)

**File:** `static/js/components/CollaborativeSandbox.js`

Full-featured code execution environment with real-time collaboration.

### Features

1. **Multi-Tab Interface**
   - Editor: Code editor with file tree
   - Terminal: Command execution
   - Output: Execution results
   - Files: Workspace browser

2. **Collaboration Mode**
   - Toggle AetherMind editing
   - Visual cursor indicators
   - Real-time code suggestions

3. **Execution**
   - Multi-language support
   - Terminal commands
   - Output/error display
   - Execution timing

### Usage

```javascript
import { CollaborativeSandbox } from './components/CollaborativeSandbox.js';

const sandbox = new CollaborativeSandbox('sandbox-container');

// Load file
sandbox.loadFile('example.py', 'print("Hello, World!")');

// Enable collaboration
sandbox.toggleCollaboration();

// Execute code
sandbox.executeCode();

// AetherMind edits (when collaboration enabled)
sandbox.aetherEdit({
    newContent: 'def improved():\n    return "Better"'
});
```

### API Endpoints

**Execute Code:**
```
POST /api/sandbox/execute
{
    "code": "...",
    "language": "python",
    "filename": "main.py"
}
```

**Terminal Command:**
```
POST /api/sandbox/terminal
{
    "command": "ls -la"
}
```

---

## Component #5: Multi-Agent Collaboration View (Advanced Feature #11)

**File:** `static/js/components/MultiAgentView.js`

Visualizes coordination between multiple specialized AI agents.

### Default Agents

1. **Code Specialist**: Programming, debugging, architecture
2. **Research Specialist**: Analysis, literature review, data
3. **Business Specialist**: Strategy, finance, marketing
4. **Orchestrator**: Meta-reasoning, coordination, synthesis

### Features

1. **Agent Cards**
   - Status indicators (idle, active, processing)
   - Confidence meters
   - Expertise tags
   - Current task display

2. **Communication Flow**
   - Canvas-based visualization
   - Animated message flows
   - Color-coded agreement:
     - Green: Agreement (>70%)
     - Yellow: Discussion (40-70%)
     - Red: Disagreement (<40%)

3. **Consensus Tracking**
   - Overall consensus percentage
   - Per-agent confidence
   - Modal detailed analysis

4. **Handoff Tracker**
   - Control transfer timeline
   - Agent-to-agent flow
   - Handoff reasons

### Usage

```javascript
import { MultiAgentView } from './components/MultiAgentView.js';

const agentView = new MultiAgentView('agent-container');

// Update agent status
agentView.updateAgentStatus('code_specialist', 'active', 'Refactoring module...');

// Add communication
agentView.addCommunication(
    'code_specialist',
    'research_specialist',
    'Need architectural patterns research',
    0.85  // High agreement
);

// Show consensus
agentView.showConsensusModal();
```

---

## Component #6: Differential Learning Visualizer (Advanced Feature #12)

**File:** `static/js/components/DifferentialLearningVisualizer.js`

Shows before/after knowledge comparison and learning progression.

### Features

1. **Split Comparison View**
   - Before learning state
   - After learning state
   - Animated transitions
   - Change highlighting

2. **Knowledge Delta Metrics**
   - Added nodes
   - Removed nodes
   - Modified nodes
   - Strengthened/weakened confidence
   - Visual delta cards

3. **Memory Promotion Pipeline**
   - Episodic → Semantic → Core
   - Stage counts
   - Flow visualization
   - Promotion tracking

4. **Gradient Flow Animation**
   - Canvas-based wave animations
   - Color-coded by learning type:
     - Green: New knowledge
     - Red: Corrections
     - Blue: Refinements
   - Fading trails

5. **Learning History Timeline**
   - Chronological event log
   - Event type indicators
   - Promotion badges
   - Detailed view buttons

### Learning Event Types

- **new**: Brand new knowledge added
- **correction**: Previous understanding corrected
- **refinement**: Existing knowledge improved

### Usage

```javascript
import { DifferentialLearningVisualizer } from './components/DifferentialLearningVisualizer.js';

const diffViz = new DifferentialLearningVisualizer('differential-container');

// Record learning event
diffViz.recordLearning(
    { concept: 'Python loops', confidence: 0.6 },  // Before
    { concept: 'Python loops', confidence: 0.95 }, // After
    {
        type: 'refinement',
        promoted: false,
        stage: 'episodic'
    }
);

// Show specific comparison
diffViz.showComparison(learningEvent);

// Calculate delta
const delta = diffViz.calculateDelta(beforeState, afterState);
```

---

## Complete Integration Example

### index.html Container Setup

```html
<!-- Activity Feed (horizontal strip at top) -->
<div id="activity-feed-container" class="activity-feed-strip"></div>

<!-- Main Chat Container -->
<div class="chat-container">
    <!-- Left: Chat Messages -->
    <div class="chat-messages" id="chat-messages"></div>
    
    <!-- Right: Visualizations -->
    <div class="visualization-panel">
        <!-- Streaming Accordion (replaces Thinking Process) -->
        <div id="accordion-container"></div>
        
        <!-- Tabs for Advanced Features -->
        <div class="viz-tabs">
            <button data-tab="knowledge-graph">Knowledge Graph</button>
            <button data-tab="sandbox">Sandbox</button>
            <button data-tab="agents">Multi-Agent</button>
            <button data-tab="learning">Differential Learning</button>
        </div>
        
        <!-- Tab Content -->
        <div class="viz-content">
            <div id="knowledge-graph-container" class="viz-panel active"></div>
            <div id="sandbox-container" class="viz-panel"></div>
            <div id="agent-container" class="viz-panel"></div>
            <div id="differential-container" class="viz-panel"></div>
        </div>
    </div>
</div>
```

### main.js Complete Setup

```javascript
import { StreamingAccordion } from './components/StreamingAccordion.js';
import { LiveActivityFeed } from './components/LiveActivityFeed.js';
import { KnowledgeGraph } from './components/KnowledgeGraph.js';
import { CollaborativeSandbox } from './components/CollaborativeSandbox.js';
import { MultiAgentView } from './components/MultiAgentView.js';
import { DifferentialLearningVisualizer } from './components/DifferentialLearningVisualizer.js';

// Initialize ALL 6 components
const accordion = new StreamingAccordion('accordion-container');
const activityFeed = new LiveActivityFeed('activity-feed-container');
const knowledgeGraph = new KnowledgeGraph('knowledge-graph-container');
const sandbox = new CollaborativeSandbox('sandbox-container');
const agentView = new MultiAgentView('agent-container');
const diffViz = new DifferentialLearningVisualizer('differential-container');

// Make globally accessible
window.accordion = accordion;
window.activityFeed = activityFeed;
window.knowledgeGraph = knowledgeGraph;
window.sandbox = sandbox;
window.multiAgentView = agentView;
window.differentialVisualizer = diffViz;

// Enhanced chat handler with ALL visualizations
async function handleChatMessage(userMessage) {
    // 1. Start cognitive process accordion
    const cognitivePanel = accordion.createPanel(
        'cognitive-process',
        'Cognitive Process',
        'fas fa-brain',
        'Analyzing your request...'
    );
    
    // 2. Add activity to feed (user's favorite!)
    const thinkingActivity = activityFeed.addActivity(
        'conversation',
        'Processing user request',
        { message: userMessage }
    );
    
    // 3. Capture knowledge snapshot for differential learning
    const beforeKnowledge = await getKnowledgeSnapshot();
    
    // 4. Make API call with EventSource for streaming
    const eventSource = new EventSource(
        `/api/chat?message=${encodeURIComponent(userMessage)}&api_key=${API_KEY}`
    );
    
    eventSource.addEventListener('reasoning', (e) => {
        const data = JSON.parse(e.data);
        accordion.streamToPanel('cognitive-process', data.text);
    });
    
    eventSource.addEventListener('memory_query', (e) => {
        const data = JSON.parse(e.data);
        
        // Add memory query activity
        const memActivity = activityFeed.addActivity('knowledge_query', 'Querying memories', data);
        
        // Update knowledge graph
        knowledgeGraph.updateFromMemory(data.results);
        
        activityFeed.updateActivity(memActivity, 'complete', `Found ${data.results.length} memories`);
    });
    
    eventSource.addEventListener('emotion', (e) => {
        const data = JSON.parse(e.data);
        accordion.appendEmotion('cognitive-process', data.name, data.intensity, data.color);
    });
    
    eventSource.addEventListener('agent_communication', (e) => {
        const data = JSON.parse(e.data);
        agentView.addCommunication(data.from, data.to, data.message, data.agreement);
    });
    
    eventSource.addEventListener('code_execution', (e) => {
        const data = JSON.parse(e.data);
        sandbox.loadFile(data.filename, data.code);
        sandbox.executeCode();
    });
    
    eventSource.addEventListener('response', (e) => {
        const data = JSON.parse(e.data);
        
        // Display message in chat
        appendMessage('aether', data.message);
        
        // Close cognitive process
        accordion.closePanel('cognitive-process', 'success', 'Response generated');
        
        // Update activity
        activityFeed.updateActivity(thinkingActivity, 'complete', 'Response delivered');
    });
    
    eventSource.addEventListener('complete', async (e) => {
        eventSource.close();
        
        // 5. Capture after state and show differential learning
        const afterKnowledge = await getKnowledgeSnapshot();
        
        if (JSON.stringify(beforeKnowledge) !== JSON.stringify(afterKnowledge)) {
            diffViz.recordLearning(beforeKnowledge, afterKnowledge, {
                type: 'refinement',
                promoted: false,
                stage: 'episodic'
            });
            
            activityFeed.addActivity('learning', 'Knowledge updated', {
                changes: Object.keys(diffViz.calculateDelta(beforeKnowledge, afterKnowledge)).length
            });
        }
    });
    
    eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        eventSource.close();
        accordion.closePanel('cognitive-process', 'error', 'Error occurred');
        activityFeed.updateActivity(thinkingActivity, 'error', 'Failed to process request');
    };
}

async function getKnowledgeSnapshot() {
    const response = await fetch('/api/knowledge/snapshot', {
        headers: { 'Authorization': `ApiKey ${API_KEY}` }
    });
    return await response.json();
}
```

---

## CSS Requirements Summary

All components require comprehensive CSS styling. Key files needed:

1. `streaming-accordion.css` - Accordion panels, status indicators, animations
2. `activity-feed.css` - Horizontal feed, sliding cards, modal views
3. `knowledge-graph.css` - Canvas container, node inspector, controls
4. `collaborative-sandbox.css` - Tabs, editor, terminal, output panels
5. `multi-agent-view.css` - Agent cards, communication flow, consensus view
6. `differential-learning.css` - Comparison panels, delta cards, gradient canvas, timeline

---

## Next Steps

### 1. Create CSS Files
Create comprehensive stylesheets for all 6 components in `static/css/` directory.

### 2. Update index.html
Add container elements for all components with proper structure.

### 3. Backend API Integration
Implement Server-Sent Events (SSE) endpoints in `orchestrator/main_api.py`:
- `/api/chat` - Streaming chat with cognitive events
- `/api/knowledge/snapshot` - Knowledge state capture
- `/api/sandbox/execute` - Code execution
- `/api/sandbox/terminal` - Terminal commands

### 4. Testing
- Component initialization
- Event streaming
- Visual updates
- Cross-component communication
- Error handling

### 5. Polish
- Smooth animations
- Responsive design
- Loading states
- Error messages
- Performance optimization

---

## Component Status

| Component | File | Status | Advanced Feature |
|-----------|------|--------|------------------|
| Streaming Accordion | `StreamingAccordion.js` | ✅ Complete | Features 15, 13, 11, 5 |
| Live Activity Feed | `LiveActivityFeed.js` | ✅ Complete | User's Favorite! |
| Knowledge Graph | `KnowledgeGraph.js` | ✅ Complete | #9 |
| Collaborative Sandbox | `CollaborativeSandbox.js` | ✅ Complete | #10 |
| Multi-Agent View | `MultiAgentView.js` | ✅ Complete | #11 |
| Differential Learning | `DifferentialLearningVisualizer.js` | ✅ Complete | #12 |

**All 6 components fully implemented!** 🎉

Ready for CSS creation, HTML integration, and backend API wiring.
