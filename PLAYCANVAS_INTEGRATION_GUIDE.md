# AetherMind PlayCanvas Integration Guide

## Overview

AetherMind can now **autonomously build 3D games** through natural language conversation. The Brain controls PlayCanvas Editor via browser automation (Playwright), creating entities, uploading scripts, and configuring game scenes without manual clicking.

---

## Architecture

```
User: "create a cyberpunk city with 50 buildings"
  ↓
Brain (logic_engine.py)
  - Reasons about scene structure
  - Plans entity hierarchy
  - Generates action tags
  ↓
<aether-playcanvas action="create_building" project_id="1449261">
{
  "building_type": "skyscraper",
  "position": {"x": 10, "y": 0, "z": 5},
  "scale": 2
}
</aether-playcanvas>
  ↓
ActionParser (orchestrator/action_parser.py)
  - Parses XML tag
  - Extracts JSON intent
  ↓
ActionExecutor._execute_playcanvas()
  - Routes to PlayCanvasEditorAdapter
  ↓
PlayCanvasEditorAdapter (body/adapters/playcanvas_editor_adapter.py)
  - Uses Playwright to control browser
  - Clicks hierarchy panel
  - Creates entity
  - Adds components
  - Sets transform values
  ↓
PlayCanvas Editor (Web UI)
  - Entity created!
  - Scene updated
  ↓
Result flows back to Brain
  - Brain sees success
  - Continues with next building
```

---

## Deployment Modes

### 1. **Local Development (Visible Browser)**

**Best for**: Testing, debugging, watching AI work in real-time

```bash
# Set environment variables
export PLAYCANVAS_HEADLESS=false
export PLAYCANVAS_USERNAME="your_username"
export PLAYCANVAS_PASSWORD="your_password"

# Start backend
./start_backend.sh

# Browser window opens automatically when you use aether-playcanvas tag
```

**What you'll see**: Chrome window pops up, navigates to PlayCanvas, logs in, opens editor, creates entities - all automatically!

---

### 2. **Production Server (Headless Container)**

**Best for**: Cloud deployment, API service, no GUI needed

```bash
# docker-compose.yml already configured!
docker-compose up orchestrator

# Or build manually
docker build -t aethermind-universal .
docker run -p 8000:8000 \
  -e PLAYCANVAS_HEADLESS=true \
  -e PLAYCANVAS_USERNAME="username" \
  -e PLAYCANVAS_PASSWORD="password" \
  aethermind-universal
```

**Requirements**:
- Docker with Chromium dependencies
- Dockerfile includes all Playwright browser binaries
- No display/X11 needed (headless mode)

---

### 3. **Background Worker (Autonomous Game Building)**

**Best for**: Long-running tasks like "build me a complete city with 100 buildings"

```python
# User creates goal via API
POST /v1/goals/create
{
  "description": "Build a cyberpunk city in PlayCanvas project 1449261",
  "user_id": "github_user"
}

# BackgroundWorker picks it up
# Runs for 30 minutes, creating entities
# User can close browser, work continues!
# Progress saved in Supabase
```

**How it works**:
1. Goal decomposed into subtasks: "Create 100 buildings", "Setup lighting", "Add player camera"
2. Each subtask generates `<aether-playcanvas>` tags
3. Browser automation runs in background
4. User gets notification when complete

---

## Usage Examples

### Example 1: Create Single Entity

**Chat:**
```
User: "Create a player character at coordinates 0, 1, 0 in my PlayCanvas project 1449261"
```

**Brain generates:**
```xml
<think>
• User wants player entity
• Need model, camera, and rigidbody components
• Position at (0, 1, 0) for elevated spawn
</think>

<aether-playcanvas action="create_entity" project_id="1449261">
{
  "entity_name": "Player",
  "components": ["model", "camera", "rigidbody"],
  "position": {"x": 0, "y": 1, "z": 0}
}
</aether-playcanvas>
```

**What happens:**
1. Playwright opens project 1449261 editor
2. Right-clicks hierarchy panel → "New Entity"
3. Renames to "Player"
4. Clicks "Add Component" → "Model"
5. Clicks "Add Component" → "Camera"
6. Clicks "Add Component" → "Rigidbody"
7. Sets position inputs: x=0, y=1, z=0
8. Returns success to Brain

---

### Example 2: Upload Custom Script

**Chat:**
```
User: "Create a player controller script that moves the character with WASD keys"
```

**Brain generates:**
```xml
<aether-playcanvas action="upload_script" project_id="1449261">
{
  "script_name": "playerController",
  "script_content": "var PlayerController = pc.createScript('playerController'); PlayerController.prototype.initialize = function() { this.speed = 5; }; PlayerController.prototype.update = function(dt) { if (this.app.keyboard.isPressed(pc.KEY_W)) { this.entity.translate(0, 0, -this.speed * dt); } /* ...more movement code... */ };"
}
</aether-playcanvas>
```

**What happens:**
1. Clicks "Assets" panel
2. Clicks "New Asset" → "Script"
3. Names it "playerController"
4. Opens script editor (double-click)
5. Pastes generated code
6. Saves (Ctrl+S)

---

### Example 3: Complete Scene Setup

**Chat:**
```
User: "Setup a basic scene with ground, sun, and camera in project 1449261"
```

**Brain generates:**
```xml
<aether-playcanvas action="setup_scene" project_id="1449261">
{
  "scene_type": "city",
  "lighting": "day"
}
</aether-playcanvas>
```

**What happens:**
1. Creates "Ground" entity with model + collision components
2. Creates "Sun" entity with directional light
3. Creates "MainCamera" entity with camera component at (0, 5, -10)
4. Configures lighting for daytime (intensity=1.0)

---

### Example 4: Deploy AetherBridge (Auto-Connect to Backend)

**Chat:**
```
User: "Connect my game to the AetherMind backend so you can control it in real-time"
```

**Brain generates:**
```xml
<aether-playcanvas action="deploy_bridge" project_id="1449261">
{
  "api_url": "http://localhost:8000/v1/game/unity/state"
}
</aether-playcanvas>
```

**What happens:**
1. Reads [playcanvas_aether_bridge.js](playcanvas_aether_bridge.js) from disk
2. Uploads as "aetherBridge" script
3. Creates "GameManager" entity
4. Attaches aetherBridge script to GameManager
5. Configures attributes:
   - `apiUrl`: "http://localhost:8000/v1/game/unity/state"
   - `syncInterval`: 500
   - `debugMode`: true

**Result**: Game now polls backend every 500ms, receives commands from Brain, sends game state back!

---

## API Reference

### Available Actions

| Action | Description | Parameters |
|--------|-------------|------------|
| `create_entity` | Create game object in scene | `entity_name`, `components[]`, `position{x,y,z}` |
| `upload_script` | Upload JavaScript code | `script_name`, `script_content` |
| `attach_script` | Attach script to entity | `entity_name`, `script_name`, `attributes{}` |
| `create_building` | Quick building spawner | `building_type`, `position{x,y,z}`, `scale` |
| `setup_scene` | Initialize base scene | `scene_type`, `lighting` |
| `deploy_bridge` | Install AetherBridge | `api_url` |

### Component Types

Supported PlayCanvas components:
- `model` - 3D mesh rendering
- `camera` - Viewport rendering
- `light` - Illumination (directional, point, spot)
- `rigidbody` - Physics simulation
- `collision` - Collision detection
- `script` - Custom behavior
- `sound` - Audio playback

---

## Environment Variables

```bash
# Required for automated login
PLAYCANVAS_USERNAME="your_username"
PLAYCANVAS_PASSWORD="your_password"

# Deployment mode
PLAYCANVAS_HEADLESS="false"  # Local: false (visible), Production: true (headless)

# Optional: Override default API endpoint
AETHER_API_URL="http://localhost:8000/v1/game/unity/state"
```

---

## Troubleshooting

### "Browser not initialized"
```bash
# Install Playwright browsers
playwright install chromium
```

### "Login failed: Timeout"
- Check username/password are correct
- Verify login URL: https://login.playcanvas.com/
- Check network connectivity

### "Entity not found after creation"
- PlayCanvas UI may have changed selectors
- Check browser console (F12) for errors
- Run in non-headless mode to debug: `PLAYCANVAS_HEADLESS=false`

### "Docker container crashes"
```dockerfile
# Ensure Dockerfile has all dependencies:
RUN playwright install chromium
RUN playwright install-deps chromium
```

---

## Advanced: Batch Operations

Create multiple buildings at once:

```python
User: "Create a grid of 10x10 buildings in my city"

# Brain generates 100 aether-playcanvas tags in one response!
for x in range(10):
    for z in range(10):
        <aether-playcanvas action="create_building" project_id="1449261">
        {
          "building_type": "office_tower",
          "position": {"x": x*10, "y": 0, "z": z*10},
          "scale": 1.5
        }
        </aether-playcanvas>
```

All 100 tags execute sequentially, browser automation handles it!

---

## Performance Considerations

### Browser Automation Speed
- **Single entity**: ~2-3 seconds
- **Script upload**: ~5-7 seconds (includes editor open time)
- **Complete scene setup**: ~15-20 seconds
- **100 buildings**: ~5-10 minutes (batched)

### Optimization Tips
1. **Batch similar operations** - Create all entities first, then attach scripts
2. **Use background worker** - Long tasks run async, don't block chat
3. **Cache browser session** - Adapter keeps browser open between calls
4. **Headless mode** - Faster without rendering UI

---

## Security Notes

1. **Credentials**: Never commit PLAYCANVAS_USERNAME/PASSWORD to git
2. **Rate limiting**: PlayCanvas may throttle automated requests
3. **Sandbox isolation**: Browser runs in container, no host access
4. **API keys**: Use dev_mode for local, generate keys for production

---

## Future Enhancements

- [ ] Asset uploading (FBX, textures)
- [ ] Scene graph queries (list all entities)
- [ ] Visual debugging (screenshot scene)
- [ ] Multi-project support
- [ ] Collaboration (multiple users building same scene)

---

## Example: Full City Build

```bash
User: "Build me a cyberpunk city with 50 skyscrapers, neon lights, flying cars, and a player that can explore"

# Brain decomposes into:
1. Setup scene (ground, sky, lighting)
2. Create 50 buildings in grid layout
3. Add neon light entities at random positions
4. Create flying car entities with AI movement scripts
5. Create player with first-person camera
6. Deploy AetherBridge for real-time control
7. Add ambient audio

# Total time: ~30 minutes (background worker)
# User gets notification when complete!
```

---

## Integration with Other Body Adapters

PlayCanvas works alongside other adapters:

```python
# Vision system analyzes game scene
<aether-research query="best practices for cyberpunk lighting"></aether-research>

# Smart home controls game environment
<aether-playcanvas action="set_lighting" ...>

# Automotive adapter generates vehicle AI
<aether-forge tool="vehicle_controller" ...>
<aether-playcanvas action="upload_script" ...>
```

**Unified cognitive loop**: Brain reasons, Playwright executes, game responds!

---

## Contact & Support

- Documentation: This file
- Code: [body/adapters/playcanvas_editor_adapter.py](body/adapters/playcanvas_editor_adapter.py)
- Bridge Script: [playcanvas_aether_bridge.js](playcanvas_aether_bridge.js)
- Issues: Check terminal output, browser console (F12)
