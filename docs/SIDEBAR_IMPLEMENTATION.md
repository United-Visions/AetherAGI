# Dynamic Sidebar Implementation Guide

## Overview
The sidebar has been completely redesigned as a collapsible, dynamic navigation system with multiple views for managing repositories, tools, and autonomous goals.

## Architecture

### Components Implemented

1. **HTML Structure** (`frontend_flask/templates/index.html`)
   - Sidebar toggle button (top-left, fixed position)
   - Sidebar overlay (click-outside-to-close)
   - Dynamic multi-view container system
   - Three main views: Main Menu, Repos, Tools, Goals

2. **CSS Styling** (`frontend_flask/static/css/sidebar.css`)
   - Collapsible slide-in animation (position: fixed, left: -280px → 0)
   - Modal overlay system for dialogs
   - Professional card-based layouts for items
   - Responsive search bars and buttons
   - Smooth transitions and hover effects

3. **JavaScript Logic** (`frontend_flask/static/js/components/Sidebar.js`)
   - Sidebar class with navigation history stack
   - View management (open, close, navigateTo, goBack)
   - API integration for loading repos and goals
   - Search functionality for tools and goals
   - Modal dialogs for connecting repos and creating goals

4. **Backend API Endpoints** (`orchestrator/main_api.py`)
   - `GET /api/user/repos` - Fetch user's connected repositories
   - `POST /api/user/repos/connect` - Connect a new repository
   - `DELETE /api/user/repos/{repo_full_name}` - Disconnect a repository
   - `GET /auth/github` - GitHub OAuth initiation (stub)
   - `GET /auth/github/callback` - GitHub OAuth callback (stub)

## Features

### 1. Collapsible Sidebar
- **Default State**: Closed
- **Toggle Button**: Fixed top-left hamburger menu
- **Close Methods**: 
  - Click toggle button
  - Click outside sidebar (on overlay)
  - Click close button in header
  - Press Escape key (handled by Sidebar.js)

### 2. Navigation System
- **Back Button**: Returns to previous view in history stack
- **View Transitions**: Smooth slide animations between views
- **Header Updates**: Icon and title change based on current view

### 3. Main Menu View
- **Repos Button**: Navigate to repositories list
- **Tools Button**: Navigate to tools browser
- **Goals Button**: Navigate to autonomous goals
- **Profile Section**: (Future enhancement)

### 4. Repositories View
- **List Display**: Shows all connected GitHub repos
- **Repo Cards**: Name, description, public/private badge
- **Add Button (+)**: Opens modal to connect new repository
- **Connect Options**: Public or private repo input
- **Remove Action**: Click repo card to disconnect

### 5. Tools View
- **Search Bar**: Filter tools by name or description
- **Core Tools Section**: Built-in MCP tools
- **My Tools Section**: User-created tools
- **Tool Cards**: Icon, name, description, type badge (MCP/PyPI)
- **Add Tool Button**: Opens toolforge interface (future)

### 6. Goals View
- **Search Bar**: Filter goals by description
- **Goal Cards**: Description, status badge, progress indicator
- **Create Button (+)**: Opens modal to create new autonomous goal
- **Goal Details**: Click card to view subtasks and progress

## API Integration

### Repository Management

#### Fetch User Repos
```javascript
GET /api/user/repos
Authorization: Bearer {token}

Response:
{
  "repos": [
    {
      "name": "repo-name",
      "full_name": "owner/repo-name",
      "description": "Repo description",
      "private": false,
      "html_url": "https://github.com/owner/repo",
      "connected_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

#### Connect New Repo
```javascript
POST /api/user/repos/connect
Authorization: Bearer {token}
Content-Type: application/json

Body:
{
  "name": "repo-name",
  "full_name": "owner/repo-name",
  "description": "Repo description",
  "private": false,
  "html_url": "https://github.com/owner/repo"
}

Response:
{
  "message": "Repository connected successfully",
  "repo": {...},
  "repos": [...]
}
```

#### Disconnect Repo
```javascript
DELETE /api/user/repos/owner%2Frepo-name
Authorization: Bearer {token}

Response:
{
  "message": "Repository disconnected successfully",
  "repos": [...]
}
```

### GitHub OAuth (Stub)
```javascript
GET /auth/github
// Returns setup instructions for GitHub OAuth App

GET /auth/github/callback?code={code}
// Handles OAuth callback (not yet implemented)
```

### Goals Integration
Sidebar integrates with existing autonomous goals system:
- `GET /v1/goals/list` - Fetch user goals
- `POST /v1/goals/create` - Create new goal
- `GET /v1/goals/{id}/status` - Check goal progress

## User Flows

### Connecting a Repository

1. User clicks sidebar toggle button
2. Sidebar slides in from left
3. User clicks "Repos" in main menu
4. Repos view displays with connected repos list
5. User clicks + button
6. Modal appears with input field
7. User enters `owner/repo-name` or GitHub URL
8. User selects Public/Private
9. User clicks "Connect Repository"
10. Frontend calls `POST /api/user/repos/connect`
11. Backend saves to Supabase user metadata
12. Modal closes, repo appears in list

### Creating an Autonomous Goal

1. User navigates to Goals view
2. User clicks + button
3. Modal appears with goal form
4. User enters description and priority
5. User clicks "Create Goal"
6. Frontend calls `POST /v1/goals/create`
7. Backend creates goal in Supabase
8. BackgroundWorker picks up goal (polls every 30s)
9. AutonomousAgent decomposes into subtasks
10. Goal appears in list with progress indicator

### Browsing Tools

1. User navigates to Tools view
2. Core Tools section shows built-in MCP tools
3. My Tools section shows user-created tools
4. User types in search bar
5. Tools filter in real-time
6. User clicks tool card to view details
7. (Future: Opens toolforge editor)

## Data Storage

### User Metadata in Supabase
```json
{
  "user_id": "uuid",
  "email": "user@example.com",
  "metadata": {
    "connected_repos": [
      {
        "name": "repo-name",
        "full_name": "owner/repo-name",
        "description": "...",
        "private": false,
        "html_url": "...",
        "connected_at": "2024-01-15T10:30:00Z"
      }
    ],
    "custom_tools": [],
    "preferences": {}
  }
}
```

## CSS Variables Used
```css
--bg-primary: Main background color
--bg-secondary: Secondary background (cards)
--bg-tertiary: Tertiary background (hover states)
--text-primary: Primary text color
--text-muted: Muted/secondary text
--border-color: Border color for elements
--accent-color: Primary accent (green)
--accent-hover: Accent hover state
```

## Browser Compatibility
- Modern browsers (Chrome, Firefox, Safari, Edge)
- CSS Grid and Flexbox required
- ES6+ JavaScript features (async/await, arrow functions)
- CSS animations and transitions

## Performance Considerations
- Sidebar renders on page load (hidden by default)
- API calls only when views are opened
- Search filtering happens client-side (no API calls)
- Modal overlays use CSS animations (GPU-accelerated)
- Click-outside detection uses event delegation

## Security Notes
- All API endpoints require Bearer token authentication
- Repository connections stored in user's own metadata
- No direct GitHub API access from frontend
- OAuth flow will be server-side (when implemented)

## Future Enhancements

### Phase 1 (Immediate)
- [ ] Apply Supabase goals schema
- [ ] Test end-to-end goal creation flow
- [ ] Add loading states for API calls
- [ ] Implement error toasts/notifications

### Phase 2 (Short-term)
- [ ] Full GitHub OAuth implementation
- [ ] Fetch repos directly from GitHub API
- [ ] Repository file browser
- [ ] Tool editor integration (Toolforge)
- [ ] Profile settings view

### Phase 3 (Long-term)
- [ ] Real-time goal progress updates (WebSocket)
- [ ] Collaborative repo sharing
- [ ] Tool marketplace
- [ ] Advanced search with filters
- [ ] Keyboard shortcuts (Ctrl+K command palette)

## Testing Checklist

### Frontend
- [ ] Sidebar opens/closes smoothly
- [ ] Click outside closes sidebar
- [ ] Back button navigates correctly
- [ ] Search filters work in tools and goals
- [ ] Modals appear centered with backdrop
- [ ] Repo cards display correctly
- [ ] Goal cards show progress

### Backend
- [ ] GET /api/user/repos returns empty array initially
- [ ] POST /api/user/repos/connect saves to Supabase
- [ ] DELETE /api/user/repos/{name} removes repo
- [ ] Authentication works on all endpoints
- [ ] Error responses return proper status codes

### Integration
- [ ] Repos persist across page refreshes
- [ ] Goals integrate with autonomous system
- [ ] Core tools load from curated_tool_index.json
- [ ] User can create and view goals
- [ ] Repo connections survive server restarts

## Troubleshooting

### Sidebar not appearing
- Check browser console for JavaScript errors
- Verify Sidebar.js is loaded (check Network tab)
- Check CSS is loaded (inspect element)

### API calls failing
- Verify backend is running
- Check Bearer token in localStorage
- Look at Network tab for 401/500 errors
- Check CORS configuration

### Modals not showing
- Verify modal CSS is loaded
- Check z-index conflicts
- Inspect modal-overlay element

### Repos not saving
- Check Supabase connection
- Verify users table has metadata JSONB column
- Check backend logs for errors

## Code References

### Key Files
- `frontend_flask/templates/index.html` (lines 49-161)
- `frontend_flask/static/css/sidebar.css` (397 lines)
- `frontend_flask/static/js/components/Sidebar.js` (389 lines)
- `orchestrator/main_api.py` (lines 845-1000)

### Key Functions
- `Sidebar.open()` - Opens sidebar with animation
- `Sidebar.navigateTo(view)` - Changes active view
- `Sidebar.loadUserRepos()` - Fetches repos from API
- `connectGitHubRepo(fullName, isPrivate)` - Connects repo
- `submitGoal(description, priority)` - Creates autonomous goal

## Deployment Notes

### Frontend
- No build step required (vanilla JS)
- Ensure static files are served correctly
- Check Flask static_folder configuration

### Backend
- New endpoints added to main_api.py
- No database migrations required (uses existing metadata column)
- OAuth endpoints are stubs (return 501 or info message)

### Environment Variables
No new variables required for basic functionality.

For full GitHub OAuth (future):
```bash
GITHUB_CLIENT_ID=your_github_app_client_id
GITHUB_CLIENT_SECRET=your_github_app_client_secret
GITHUB_REDIRECT_URI=https://your-domain.com/auth/github/callback
```

## Success Metrics

After implementation, verify:
1. ✅ Sidebar opens/closes without page reload
2. ✅ Navigation works between all views
3. ✅ Search functionality filters items correctly
4. ✅ Repos can be connected and stored
5. ✅ Goals can be created from sidebar
6. ✅ UI is responsive on different screen sizes
7. ✅ No console errors in browser
8. ✅ API endpoints return proper responses

---

**Implementation Status**: ✅ Complete (Frontend + Backend Stubs)

**Next Steps**: 
1. Apply Supabase goals schema
2. Test autonomous goal creation
3. Implement full GitHub OAuth
4. Add loading states and error handling
