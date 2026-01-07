# AetherMind Shell UI Architecture

## Overview

The Shell UI is a **minimal, agent-controlled interface** where the agent has REAL conversations with users. No simulated responses - every message comes from the actual AI backend.

## Key Principles

1. **Real Conversations**: Agent talks via the actual LLM, not scripted responses
2. **Organic Learning**: Agent discovers user needs through natural dialogue, no forced domains
3. **Step-by-Step Persistence**: Every exchange is saved incrementally to user profile
4. **Minimal Shell**: Only chat and notifications visible by default
5. **Background-First**: All complex features run as background tasks (like Google's Jules)
6. **Pilot Users**: Admins (in `settings.yaml`) see additional controls like benchmarks

## Onboarding Flow

```
User visits /chat
     ↓
OnboardingAgent.start()
     ↓
Sends to backend with context: { isOnboarding: true, mode: "onboarding_start" }
     ↓
Backend injects special system prompt for warm, curious personality
     ↓
Agent introduces herself, asks user's name
     ↓
User responds → saved to profile.conversationLog
     ↓
Agent responds naturally, learns facts organically
     ↓
Backend parses response for [ONBOARDING_COMPLETE] or [REQUEST_MEDIA] markers
     ↓
When complete, profile.onboarded = true, regular chat begins
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Shell Container                          │
├─────────────────────────────────────────────────────────────────┤
│  Header: Brand + Notifications + Admin (pilot only)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      Chat Area                                  │
│                  (Messages Container)                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Dynamic Slots (Agent injects components here)          │   │
│  │  - Panels, forms, visualizations as needed              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Input Area: Quick Actions (hidden) + Text Input + Send        │
├─────────────────────────────────────────────────────────────────┤
│  Background Tasks Indicator (bottom corner)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Files Created

### Templates
- `templates/index_shell.html` - Minimal shell template

### CSS
- `static/css/shell.css` - All shell UI styles

### JavaScript Core
- `static/js/shell-router.js` - Main shell entry point
- `static/js/core/OnboardingAgent.js` - Interactive user onboarding
- `static/js/core/UIOrchestrator.js` - Agent-controlled component injection
- `static/js/core/NotificationManager.js` - Notification system
- `static/js/core/BackgroundTaskManager.js` - Jules-style persistent tasks

### Backend Updates
- `orchestrator/main_api.py` - Added user profile & task endpoints
- `frontend_flask/app.py` - Updated to serve shell, pilot user check
- `config/settings.yaml` - Added shell UI configuration

## User Flow

### First Visit (Onboarding)
1. User sees minimal chat with greeting from AetherMind
2. Agent asks for name → purpose → work style → optional photo
3. User can decline any question ("I'll ask again later")
4. Profile saved, features unlocked based on needs

### Returning User
1. Agent greets by name with time-appropriate message
2. Quick actions shown based on previous usage
3. Background tasks resume from where they left off

### Agent Feature Activation
The agent can show/hide features via `ui_commands` in responses:
```javascript
{
  "ui_commands": [
    { "action": "show_quick_actions" },
    { "action": "show_component", "component": "camera" },
    { "action": "inject_panel", "html": "...", "position": "dynamic" }
  ]
}
```

## Pilot User Features

Users in `settings.yaml` → `pilot_users` array see:
- Admin panel button in header
- Benchmark runner controls
- Detailed task management
- User profile viewer

```yaml
pilot_users: ["United-Visions", "your-github-username"]
```

## Background Task Types

All complex operations run as persistent background tasks:

| Type | Description |
|------|-------------|
| `research` | Web research, knowledge gathering |
| `build` | App/tool creation in sandbox |
| `tool_forge` | Creating new agent capabilities |
| `goal` | Autonomous multi-step goals |
| `mcp_setup` | Adding MCP servers |
| `benchmark` | Running performance benchmarks |

## Tool Forging Priority

From `settings.yaml`:
```yaml
tool_forging:
  enabled: true
  priority: secondary  # Fallback after search
  fallback_after_search: true
```

When firecrawl/search returns poor results, agent automatically forges custom tools.

## Endpoints

### User Profile
- `GET /v1/user/profile` - Get user profile & onboarding status
- `POST /v1/user/profile` - Save profile from onboarding

### Background Tasks
- `POST /v1/tasks/status` - Get status of running tasks
- `POST /v1/tasks/create` - Create new background task

### Legacy
- `/chat/legacy` - Full UI with all features visible (for debugging)

## Migration from Old UI

The old `index.html` template with sidebar, brain visualizer, etc. is preserved at `/chat/legacy`. The new shell at `/chat` is the default experience.

All existing functionality (sandbox, tool forge, benchmarks) still works—it just runs in the background instead of visible UI panels.
