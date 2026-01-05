# Quick Start Guide - Testing the New AGI Frontend

## Prerequisites

- Flask server running
- Browser with JavaScript enabled
- API keys configured in `.env`

## Launch Steps

### 1. Start the Frontend

```bash
cd /Users/deion/Desktop/aethermind_universal/frontend_flask
python app.py
```

### 2. Open Browser

Navigate to: `http://localhost:5000`

### 3. Test the New Features

#### A. Activity Feed Toggle
1. Look for the **stream icon** (📊) in the top-right corner
2. Click it to slide open the activity feed from the right
3. The feed shows all agent activities in real-time

#### B. Send a Message
1. Type a message in the chat input
2. Press Enter or click Send
3. **Observe:**
   - Activity feed logs "Processing your request" (in_progress)
   - Brain visualizer (if open) shows 6-stage thinking animation
   - Old thinking visualizer shows card in left sidebar
   - Activity feed updates to "completed" when done
   - New activity appears: "Interaction saved to episodic memory"

#### C. Brain Visualizer
1. Look for the **brain icon** (🧠) in the bottom-right corner
2. Click it to reveal the brain visualizer at bottom of screen
3. Send a message and watch:
   - 6 stages light up sequentially:
     1. **Sense** → Parse intent
     2. **Retrieve** → Search memory
     3. **Reason** → Apply logic
     4. **Embellish** → Add emotion
     5. **Act** → Generate response
     6. **Learn** → Store memory
   - Metrics update (surprise score, confidence, response time)
   - Neural network animation flows

#### D. Split View Panel (Detailed Inspection)
1. Click on **any activity card** in the activity feed
2. A panel slides in from the left covering 50% of screen
3. Explore the **5 tabs:**
   - **Overview**: Summary and status
   - **Code**: Syntax-highlighted source code
   - **Diff**: File changes (what was added/removed)
   - **Preview**: Live preview of running apps (iframe)
   - **Environment**: Dependencies and system info
4. Click **X** or outside panel to close

#### E. Upload a File
1. Click the **paperclip icon** (📎) in the chat input area
2. Select an image or document
3. **Observe:**
   - Activity feed logs "Uploading [filename]" (in_progress)
   - Old visualizer shows upload card
   - Activity updates to completed with analysis results
   - If surprise score > 0.5, a new "High novelty detected!" activity appears

## Console Testing (Advanced)

Open browser DevTools (F12) and run these commands:

```javascript
// Add a test activity
window.activityFeed.addActivity({
    id: 'demo_tool_creation',
    type: 'tool_creation',
    status: 'completed',
    title: 'Created Web Scraper Tool',
    details: 'Successfully built BeautifulSoup-based scraper',
    timestamp: new Date().toISOString(),
    data: {
        code: `def scrape_website(url):
    import requests
    from bs4 import BeautifulSoup
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.find_all('a')`,
        files: ['tools/web_scraper.py'],
        diff: `+++ tools/web_scraper.py
@@ -0,0 +1,7 @@
+def scrape_website(url):
+    import requests
+    from bs4 import BeautifulSoup
+    
+    response = requests.get(url)
+    soup = BeautifulSoup(response.content, 'html.parser')
+    return soup.find_all('a')`,
        environment: {
            dependencies: ['beautifulsoup4==4.12.0', 'requests==2.31.0'],
            python_version: '3.11.0',
            runtime: 'local'
        },
        preview_url: null
    }
});

// Start brain thinking demo
window.brainViz.startThinking();

// Update metrics after 2 seconds
setTimeout(() => {
    window.brainViz.updateMetrics({
        surprise_score: 0.45,
        confidence: 0.92,
        response_time: 1850
    });
}, 2000);

// Stop thinking after 5 seconds
setTimeout(() => {
    window.brainViz.stopThinking();
}, 5000);
```

## Visual Checklist

### ✅ Activity Feed
- [ ] Toggle button visible in top-right
- [ ] Feed slides in from right (300px width)
- [ ] Activity cards show type icons
- [ ] Status indicators pulse for "in_progress"
- [ ] Clicking card opens split view

### ✅ Split View Panel
- [ ] Panel slides in from left (50% width)
- [ ] All 5 tabs are clickable
- [ ] Overview tab shows summary
- [ ] Code tab has syntax highlighting
- [ ] Diff tab shows +/- changes
- [ ] Preview tab has iframe (if preview_url exists)
- [ ] Environment tab shows dependencies
- [ ] Close button works

### ✅ Brain Visualizer
- [ ] Toggle button visible in bottom-right
- [ ] Visualizer expands from bottom (250px height)
- [ ] 6 stage indicators visible
- [ ] Stages light up during thinking
- [ ] Metrics grid shows 3 values
- [ ] Neural network animation renders
- [ ] Canvas updates at 60fps

### ✅ Domain Indicator
- [ ] Header shows domain name
- [ ] Icon matches domain type
- [ ] Color matches domain theme
- [ ] Updates when domain changes

## Expected Behavior

### Normal Message Flow

1. **User types message** → Press Enter
2. **Chat shows user bubble**
3. **Activity Feed logs:**
   - "Processing your request" (in_progress)
4. **Brain Visualizer:**
   - Stage 1 (Sense) activates
   - Stage 2 (Retrieve) activates
   - Stage 3 (Reason) activates
   - Stage 4 (Embellish) activates
   - Stage 5 (Act) activates
   - Stage 6 (Learn) activates
5. **API returns response**
6. **Activity Feed updates:**
   - "Processing your request" (completed)
7. **Activity Feed logs:**
   - "Interaction saved to episodic memory" (completed)
8. **Brain Visualizer:**
   - Metrics update
   - Animation stops
9. **Chat shows assistant bubble**

### File Upload Flow

1. **User selects file** → Click paperclip, choose file
2. **Activity Feed logs:**
   - "Uploading [filename]" (in_progress)
3. **API processes file**
4. **Activity Feed updates:**
   - "Uploading [filename]" (completed)
   - Details show analysis and surprise score
5. **If surprise > 0.5:**
   - "High novelty detected!" (completed)
6. **Chat shows file attachment + assistant response**

## Troubleshooting

### Issue: Components not loading
**Solution:** Check browser console for import errors. Ensure all 3 component files exist:
- `ActivityFeed.js`
- `SplitViewPanel.js`
- `BrainVisualizer.js`

### Issue: Styling looks broken
**Solution:** Verify CSS files are imported in `main.css`:
```css
@import './activity-feed.css';
@import './split-view.css';
@import './brain-visualizer.css';
```

### Issue: Toggle buttons don't work
**Solution:** Check that HTML elements exist in `index.html`:
- `<button id="activity-feed-toggle">`
- `<button id="brain-visualizer-toggle">`
- `<div id="activity-feed-container">`
- `<div id="split-view-container">`
- `<div id="brain-visualizer-container">`

### Issue: Activities not appearing
**Solution:** Check that `activityFeed.addActivity()` is being called in main.js. Verify in console:
```javascript
console.log(window.activityFeed);
```

### Issue: Brain visualizer not animating
**Solution:** Ensure canvas element is created and `requestAnimationFrame` is running. Check console for canvas errors.

## Next Steps

### Backend Integration

To fully connect these components, enhance backend responses in `orchestrator/main_api.py`:

```python
# Add to /v1/chat/completions response
response = {
    "choices": [...],
    "metadata": {
        "agent_state": {
            "surprise_score": float,  # 0.0 to 1.0
            "confidence": float        # 0.0 to 1.0
        },
        "reasoning_steps": list,      # List of strings
        "timing": {
            "total_ms": int
        }
    }
}
```

### WebSocket Support (Future)

For real-time activity streaming during long operations:

```python
@app.websocket("/v1/ws/activities/{user_id}")
async def activity_stream(websocket: WebSocket, user_id: str):
    await websocket.accept()
    # Emit activities as agent works
    await websocket.send_json({
        "type": "activity",
        "data": {...}
    })
```

## Success Criteria

✅ Activity feed shows all agent operations  
✅ Clicking activities opens detailed split view  
✅ Brain visualizer shows 6-stage thinking process  
✅ Code tab displays syntax-highlighted source  
✅ Diff tab shows file changes  
✅ Preview tab can display running apps  
✅ Environment tab shows dependencies  
✅ Smooth animations and transitions  
✅ Domain indicator reflects user's selection  
✅ No JavaScript errors in console  

---

**Congratulations!** You've successfully implemented a sophisticated AGI frontend that provides complete transparency into AetherMind's cognitive processes with a beautiful, lovable interface. 🎉
