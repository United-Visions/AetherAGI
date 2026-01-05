# AetherMind Frontend Data Flow & Architecture

## Overview

This document provides a comprehensive understanding of how the AetherMind Flask frontend application works, including data flows, component interactions, and user journeys.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [User Authentication Flow](#user-authentication-flow)
3. [Onboarding Process](#onboarding-process)
4. [Chat Interface Data Flow](#chat-interface-data-flow)
5. [API Key Management](#api-key-management)
6. [Component Interactions](#component-interactions)
7. [Backend Integration](#backend-integration)
8. [State Management](#state-management)

---

## Architecture Overview

### Technology Stack

**Backend (Flask):**
- Flask web framework
- Session management (cookies)
- GitHub OAuth integration
- Cryptography (Fernet encryption)
- httpx for async HTTP requests

**Frontend:**
- Vanilla JavaScript (ES6 modules)
- Tailwind CSS for styling
- Font Awesome icons
- Component-based architecture

**Integration:**
- FastAPI backend (`orchestrator/main_api.py`)
- Pinecone Vector Database
- RunPod inference endpoint

### Folder Structure

```
frontend_flask/
├── app.py                      # Flask application entry point
├── templates/
│   ├── index_home.html         # Landing page
│   ├── onboarding.html         # User onboarding & key generation
│   ├── index.html              # Main chat interface
│   ├── documentation.html      # API documentation
│   ├── pricing.html            # Pricing page
│   └── domain_*.html           # Domain-specific pages
└── static/
    ├── css/
    │   └── main.css            # Modular CSS
    └── js/
        ├── router.js           # Application entry point
        ├── api.js              # Backend API communication
        └── components/
            ├── ApiKeyModal.js          # API key input modal
            ├── ChatInterface.js        # Chat UI
            ├── ThinkingVisualizer.js   # Thought process display
            ├── FileUploader.js         # File upload handling
            ├── ActivityFeed.js         # Real-time activity stream
            ├── SplitViewPanel.js       # Split view for code/docs
            └── BrainVisualizer.js      # Brain metrics visualization
```

---

## User Authentication Flow

### 1. GitHub OAuth Login

```
User clicks "Login with GitHub"
    ↓
Flask: /github_login
    ↓
Redirect to GitHub OAuth
    ↓
User authorizes application
    ↓
GitHub redirects to /callback with code
    ↓
Flask exchanges code for access_token
    ↓
Flask fetches user info from GitHub API
    ↓
Encrypt and store token in session cookie
    ↓
Store github_user in session
    ↓
Redirect to /onboarding
```

**Key Files:**
- `app.py`: `/github_login` and `/callback` routes

**Session Data:**
- `session['github_user']`: GitHub username
- `session['github_token']`: Encrypted GitHub access token

### 2. Session Management

- Sessions are stored in encrypted cookies
- Session secret key: `FLASK_SECRET` environment variable
- GitHub token encrypted with Fernet cipher
- Session persists across page reloads

---

## Onboarding Process

The onboarding flow has been redesigned with multiple steps for better UX.

### Step 1: Domain Selection

**Purpose:** User selects their AI specialization domain

**Domains Available:**
- **Software Development** (`code`): Production-ready code, debugging
- **Research & Analysis** (`research`): Academic rigor, citations
- **Business & Strategy** (`business`): Strategic frameworks, ROI focus
- **Legal** (`legal`): Case research, contracts, legal writing
- **Finance** (`finance`): Financial modeling, analysis
- **Multi-Domain Master** (`general`): Cross-disciplinary synthesis

**Data Flow:**
```
User clicks domain card
    ↓
JavaScript: domain card selected
    ↓
Update hidden input: selected-domain
    ↓
User clicks "Continue"
    ↓
Show Step 2: Login Method
```

**Key Code:**
- `templates/onboarding.html`: Domain selection cards
- JavaScript: Domain selection event listeners

### Step 2: Login Method Selection

**Purpose:** Choose how to access AetherMind

**Options:**
1. **Create New API Key**: Generate a new key for this domain
2. **Use Existing Key**: Enter a previously generated key

**Data Flow:**
```
User clicks "Create New API Key"
    ↓
Show Step 3: Create API Key
    ↓
Call generateApiKey() function
    ↓
POST /create_key with domain
    ↓
Backend generates or returns master key
    ↓
Display key to user with copy button
```

OR

```
User clicks "Use Existing Key"
    ↓
Show Step 4: Enter Existing Key
    ↓
User pastes API key
    ↓
Validate format (starts with "am_live_")
    ↓
Store in localStorage
    ↓
Redirect to chat
```

### Step 3: API Key Creation

**Purpose:** Generate and display new API key

**Backend Flow:**
```python
POST /create_key
    ↓
Check if user authenticated (session['github_user'])
    ↓
Get domain from request (JSON or form)
    ↓
Generate API key:
    - Use MASTER_API_KEY from env (development)
    - OR generate new key with AuthManager (production)
    ↓
Register domain with backend API
    ↓
Return JSON: { api_key, domain, user_id }
```

**Frontend Flow:**
```javascript
async function generateApiKey()
    ↓
Fetch POST /create_key with domain
    ↓
Receive JSON response with api_key
    ↓
Display key in <div> with copy button
    ↓
Store in localStorage:
    - aethermind_api_key
    - aethermind_domain
```

**Key Features:**
- **Copy to Clipboard**: One-click copy button with visual feedback
- **Security Warning**: Alerts user that key won't be shown again
- **Confirmation Checkbox**: User must confirm they saved the key
- **Explanation Section**: Shows what happens next (axiom seeding, memory creation, etc.)

### Step 4: Enter Existing Key

**Purpose:** Allow users with existing keys to log in

**Validation:**
- Must not be empty
- Must start with `am_live_`
- Must be at least 20 characters

**Data Flow:**
```
User pastes key in input field
    ↓
Validate format
    ↓
If valid:
    Store in localStorage
    Redirect to /chat
If invalid:
    Show error message
```

---

## Chat Interface Data Flow

### Component Initialization

```
Page Load: /chat
    ↓
Load router.js (ES6 module)
    ↓
Initialize components:
    - ApiKeyModal
    - ChatInterface
    - ThinkingVisualizer
    - FileUploader
    - ActivityFeed
    - SplitViewPanel
    - BrainVisualizer
    ↓
Check for API key in localStorage
    ↓
If no key: Show ApiKeyModal
If key exists: Continue loading
```

### Message Send Flow

**User sends a text message:**

```
User types message and presses Send
    ↓
router.js: handleSend()
    ↓
Add user message to chat UI
    ↓
Add to messageHistory array
    ↓
Create "thinking" activity in ActivityFeed
    ↓
Start brain visualizer animation
    ↓
api.js: sendMessage(messageHistory)
    ↓
Check for API key with getApiKey()
    ↓
If no key: Show ApiKeyModal
    ↓
Construct payload: { model, user, messages }
    ↓
POST to backend API endpoint
    ↓
Receive response: { choices, metadata }
    ↓
Extract assistant message
    ↓
Update ActivityFeed (thinking → completed)
    ↓
Update brain visualizer metrics
    ↓
Add assistant message to chat UI
    ↓
Log to episodic memory (backend)
```

**Detailed API Call:**

```javascript
// api.js
async sendMessage(messages) {
    const apiKey = this.getApiKey();
    
    const payload = {
        model: 'aethermind-v1',
        user: 'flask_user_01',
        messages: messages
    };
    
    const response = await fetch(this.baseUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Aether-Secret-Key': apiKey
        },
        body: JSON.stringify(payload)
    });
    
    return await response.json();
}
```

### File Upload Flow

**User attaches and sends a file:**

```
User clicks file button
    ↓
FileUploader: Open file picker
    ↓
User selects file(s)
    ↓
Preview files in UI
    ↓
User clicks Send
    ↓
For each file:
    Create upload activity in ActivityFeed
    Add file attachment to chat UI
    Create thinking card
    ↓
    api.js: uploadFile(file)
    ↓
    Create FormData with file
    ↓
    POST to /v1/ingest/multimodal
    ↓
    Backend analyzes file (perception service)
    ↓
    Returns: { analysis, surprise }
    ↓
    Update thinking card with results
    Update ActivityFeed
    ↓
    If surprise > 0.5:
        Create "high novelty" activity
    ↓
    Add assistant message with analysis
```

---

## API Key Management

### Storage

**localStorage:**
- `aethermind_api_key`: The API key string
- `aethermind_domain`: Selected domain (code, research, etc.)

### Retrieval

**api.js getApiKey():**
```javascript
getApiKey() {
    let key = localStorage.getItem('aethermind_api_key');
    
    if (!key) {
        if (apiKeyModal) {
            apiKeyModal.show();
            return null;
        } else {
            key = prompt("Please enter your API key:");
            if (key) {
                localStorage.setItem('aethermind_api_key', key);
            }
        }
    }
    
    return key;
}
```

### Modal Display

**ApiKeyModal Component:**

**When Shown:**
1. On page load if no key in localStorage
2. When API call detects missing key
3. User manually requests (future feature)

**Features:**
- Beautiful gradient background with blur
- Input validation (must start with `am_live_`)
- Error messages for invalid format
- Link to onboarding page
- Cancel button (closes modal)
- Submit button (validates and stores key)

**Validation:**
```javascript
handleSubmit() {
    const apiKey = input.value.trim();
    
    // Check: not empty
    if (!apiKey) {
        showError('Please enter your API key');
        return;
    }
    
    // Check: correct prefix
    if (!apiKey.startsWith('am_live_')) {
        showError('Invalid format');
        return;
    }
    
    // Check: minimum length
    if (apiKey.length < 20) {
        showError('Key too short');
        return;
    }
    
    // Valid: store and reload
    localStorage.setItem('aethermind_api_key', apiKey);
    window.location.reload();
}
```

---

## Component Interactions

### Component Communication

**1. router.js (Orchestrator)**
- Entry point for application
- Initializes all components
- Handles user events (send message, upload file)
- Coordinates between components

**2. api.js (Backend Client)**
- Handles all HTTP requests
- Manages API key retrieval
- Error handling and retries

**3. ChatInterface**
- Displays messages (user, assistant)
- Handles message rendering
- Shows file attachments
- Markdown rendering (if implemented)

**4. ThinkingVisualizer**
- Creates "thinking cards"
- Shows step-by-step reasoning
- Success/error states

**5. ActivityFeed**
- Real-time activity stream
- Shows: thinking, file uploads, memory updates, surprise detection
- Status tracking (in_progress, completed, error)

**6. BrainVisualizer**
- Visual representation of "brain state"
- Shows metrics: surprise, confidence, response time
- Animation during thinking

**7. ApiKeyModal**
- API key input interface
- Validation and error display
- Integration with api.js

### Event Flow Example: User Sends Message

```
User types and clicks Send
    ↓
router.js: handleSend() event handler
    ↓
chatInterface.addMessage('user', text)
    → Updates chat UI immediately
    ↓
activityFeed.addActivity({type: 'thinking'})
    → Shows "Processing..." activity
    ↓
brainViz.startThinking()
    → Starts pulsing animation
    ↓
thinkingVisualizer.createCard("Thinking Process")
    → Creates new card in UI
    ↓
api.sendMessage(messageHistory)
    → api.getApiKey() checks for key
    → If missing: apiKeyModal.show()
    → Makes HTTP request to backend
    ↓
Receive response from backend
    ↓
thinkingVisualizer.setSuccess()
    → Mark card as complete
    ↓
activityFeed.updateActivity(id, {status: 'completed'})
    → Update activity status
    ↓
brainViz.updateMetrics({surprise, confidence})
    → Update visual metrics
    ↓
chatInterface.addMessage('assistant', response)
    → Display AI response
    ↓
activityFeed.addActivity({type: 'memory_update'})
    → Log memory save
```

---

## Backend Integration

### Endpoints Used

**1. Backend API (FastAPI - orchestrator/main_api.py)**

**Endpoint:** `POST /v1/chat/completions`

**Request:**
```json
{
  "model": "aethermind-v1",
  "user": "flask_user_01",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

**Response:**
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Response text here"
      }
    }
  ],
  "metadata": {
    "agent_state": {
      "surprise_score": 0.3,
      "confidence": 0.85
    },
    "reasoning_steps": [
      "Step 1...",
      "Step 2..."
    ],
    "timing": {
      "total_ms": 1500
    }
  }
}
```

**2. Multimodal Upload**

**Endpoint:** `POST /v1/ingest/multimodal`

**Request:** FormData with file

**Response:**
```json
{
  "analysis": "Description of file content",
  "surprise": 0.7,
  "file_id": "uuid"
}
```

**3. Flask Backend (app.py)**

**Endpoint:** `POST /create_key`

**Request:**
```json
{
  "domain": "code"
}
```

**Response:**
```json
{
  "api_key": "am_live_xxxxxxxxxxxxxx",
  "domain": "code",
  "user_id": "github_username"
}
```

### Authentication

**API Key Header:**
```
Aether-Secret-Key: am_live_xxxxxxxxxxxxxx
```

**Backend Validation:**
```python
# orchestrator/auth_manager.py
def verify_api_key(provided_key: str) -> Optional[Dict]:
    key_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    key_data = self.key_store.get(key_hash)
    
    if not key_data or key_data.get("revoked"):
        return None
    
    return {
        "user_id": key_data["user_id"],
        "role": key_data["role"],
        "permissions": [...]
    }
```

---

## State Management

### Client-Side State

**localStorage:**
- `aethermind_api_key`: Persistent API key
- `aethermind_domain`: User's selected domain

**Session State (in-memory):**
- `messageHistory`: Array of chat messages
- `activityFeedOpen`: Boolean for feed visibility
- `brainVisualizerOpen`: Boolean for brain viz visibility

**Component State:**
- Each component maintains its own internal state
- Components communicate through callbacks and events

### Server-Side State

**Flask Session (encrypted cookie):**
- `github_user`: GitHub username
- `github_token`: Encrypted GitHub access token
- `user_domain`: Selected domain

**Backend State (FastAPI):**
- `SessionManager`: Manages active user sessions
- `MessageHistory`: Stored per user
- `VectorStore`: Episodic memory in Pinecone

---

## Key Improvements Made

### 1. Onboarding UX
- ✅ Multi-step flow (domain → login method → key generation/entry)
- ✅ Visual domain cards with hover effects
- ✅ API key display with copy-to-clipboard
- ✅ Security warnings and confirmation checkbox
- ✅ Option to use existing key OR create new one

### 2. API Key Management
- ✅ Beautiful modal dialog (no ugly prompt)
- ✅ Input validation with clear error messages
- ✅ Link to onboarding for new users
- ✅ Consistent styling with rest of app
- ✅ Automatic storage in localStorage

### 3. Chat View Protection
- ✅ Automatic modal display when no key found
- ✅ Non-blocking (user can cancel)
- ✅ Persistent check on each API call
- ✅ Graceful error handling

### 4. Backend Integration
- ✅ JSON endpoint for API key generation
- ✅ Support for both form and JSON requests
- ✅ Proper error handling and fallbacks
- ✅ Domain registration with backend

---

## Testing the Flow

### Test Case 1: New User (No API Key)

1. Navigate to `/chat`
2. **Expected:** ApiKeyModal appears immediately
3. User can:
   - Click "Go to Onboarding" → redirects to `/`
   - Enter existing key → validates and stores
   - Click "Cancel" → modal closes (but will reappear on next action)

### Test Case 2: Onboarding - Create New Key

1. Login with GitHub
2. Select domain (e.g., "Software Development")
3. Click "Continue"
4. Click "Create New API Key"
5. **Expected:** Key appears with copy button
6. Copy key to clipboard
7. Check "I have securely saved my API key"
8. Click "Enter Chat"
9. **Expected:** Redirected to chat with key in localStorage

### Test Case 3: Onboarding - Use Existing Key

1. Login with GitHub
2. Select domain
3. Click "Continue"
4. Click "Use Existing Key"
5. Paste key (e.g., `am_live_test123456789012345678901234567890`)
6. Click "Verify & Continue"
7. **Expected:** Redirected to chat with key stored

### Test Case 4: Chat with API Key

1. Have key in localStorage
2. Navigate to `/chat`
3. **Expected:** No modal, chat loads normally
4. Send a message
5. **Expected:** Message sent to backend with key in header
6. **Expected:** Response displayed in chat

---

## Environment Variables

### Required for Production

```env
# Flask
FLASK_SECRET=your-secret-key-here

# GitHub OAuth
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
GITHUB_REDIRECT_URI=https://your-domain.com/callback
GITHUB_REDIRECT_URI_DEV=http://127.0.0.1:5000/callback

# Encryption
FERNET_KEY=your-fernet-key-here
ENCRYPTION_KEY=your-encryption-key-here

# API Keys
MASTER_API_KEY=am_live_xxxxxxxxxxxxxx
AM_LIVE_KEY=am_live_xxxxxxxxxxxxxx

# Backend
BACKEND_API_URL=http://localhost:8000

# Auth
JWT_SECRET=your-jwt-secret-here
```

---

## Troubleshooting

### Issue: "API Key Required" modal keeps appearing

**Solution:**
- Check browser console for errors
- Verify key is stored: `localStorage.getItem('aethermind_api_key')`
- Check key format: must start with `am_live_`
- Clear localStorage and re-enter key

### Issue: "Create API Key" button doesn't work

**Solution:**
- Check if user is authenticated (`session['github_user']`)
- Check Flask logs for errors
- Verify `/create_key` endpoint is accessible
- Check network tab for request/response

### Issue: Chat messages not sending

**Solution:**
- Verify API key is present
- Check backend API is running (`http://localhost:8000`)
- Check CORS settings in `orchestrator/main_api.py`
- Verify `Aether-Secret-Key` header is being sent

---

## Future Enhancements

1. **API Key Management Dashboard**
   - View all generated keys
   - Revoke keys
   - Key usage statistics

2. **Domain Switching**
   - Allow users to switch domains without logging out
   - Different "personas" for different domains

3. **Session Persistence**
   - Save chat history across sessions
   - Resume conversations

4. **Enhanced Security**
   - Rate limiting on frontend
   - Key expiration warnings
   - Two-factor authentication

5. **Analytics**
   - Track user interactions
   - Measure API usage
   - Performance metrics

---

## Conclusion

The AetherMind frontend provides a seamless user experience for:
- **Authentication:** GitHub OAuth integration
- **Onboarding:** Multi-step domain selection and key generation
- **Chat Interface:** Real-time AI interactions with rich features
- **Security:** API key management with proper validation

The component-based architecture ensures modularity, maintainability, and scalability for future enhancements.
