# 🎮 PlayCanvas API Integration Guide

**AetherMind can now create and manage PlayCanvas games programmatically!**

---

## 🚀 What This Enables

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Auto Project Creation** | Create PlayCanvas projects from code | Generate custom demos per user |
| **Script Upload** | Deploy AetherBridge automatically | One-command setup |
| **Asset Management** | Upload scripts, JSON, shaders, HTML, CSS | Dynamic game content |
| **Scene Management** | List and manage scenes | Version control |
| **Build & Deploy** | Download built apps via API | CI/CD pipeline |

---

## 📋 Setup (2 Minutes)

### 1. Get PlayCanvas API Token

1. Go to **[playcanvas.com/account](https://playcanvas.com/account)**
2. Scroll to **"API Tokens"** section
3. Click **"Generate Token"**
4. Name it: "AetherMind Integration"
5. **Copy the token** (you won't see it again!)

### 2. Add Token to .env

```bash
# In /Users/deion/Desktop/aethermind_universal/.env
PLAYCANVAS_API_TOKEN=your_token_here
```

### 3. Test the CLI

```bash
cd /Users/deion/Desktop/aethermind_universal
python tools/playcanvas_cli.py list-projects
```

You should see your PlayCanvas projects listed!

---

## 🎯 CLI Commands

### List All Projects
```bash
python tools/playcanvas_cli.py list-projects
```

Output:
```
📦 Found 3 project(s):

  • My Game
    ID: 123456
    Created: 2026-01-01T12:00:00
    URL: https://playcanvas.com/project/123456/overview
```

### Create New Project
```bash
python tools/playcanvas_cli.py create-project "AetherMind Demo"
```

Output:
```
🚀 Creating project: AetherMind Demo...
✅ Project created successfully!
   ID: 789012
   URL: https://playcanvas.com/project/789012/overview
```

### Deploy AetherMind Bridge (Auto-Setup!)
```bash
python tools/playcanvas_cli.py deploy-aether-bridge 789012
```

This **automatically**:
1. Uploads `playcanvas_aether_bridge.js` to your project
2. Makes it available as a script component
3. Shows you next steps

Output:
```
🧠 Deploying AetherMind Bridge to project 789012...
✅ AetherMind Bridge deployed successfully!

📝 Next steps:
   1. Open project: https://playcanvas.com/project/789012/overview
   2. Create/select an entity (e.g., GameManager)
   3. Add Script component
   4. Select 'aetherBridge' script
   5. Configure API URL: http://localhost:8000/v1/game/unity/state
   6. Launch your game!
```

### Upload Custom Script
```bash
python tools/playcanvas_cli.py upload-script 789012 ./my_vehicle_ai.js
```

### Get Project Info
```bash
python tools/playcanvas_cli.py get-project 789012
```

### List Scenes
```bash
python tools/playcanvas_cli.py list-scenes 789012
```

---

## 🧠 Use via AetherMind Brain

The PlayCanvas adapter is registered in the router, so you can control it through natural language!

### Example Chat Conversation

**You:** "Create a new PlayCanvas project called 'Smart City Demo'"

**AetherMind Brain:**
```xml
<aether-route body="playcanvas">
{
  "action": "create_project",
  "params": {
    "name": "Smart City Demo",
    "description": "AI-controlled virtual city"
  }
}
</aether-route>
```

**Result:** Project created with ID returned

---

**You:** "Upload the AetherBridge script to project 789012"

**AetherMind Brain:**
```xml
<aether-route body="playcanvas">
{
  "action": "upload_script",
  "params": {
    "project_id": 789012,
    "branch_id": "master",
    "name": "aetherBridge.js",
    "content": "... [bridge script content] ..."
  }
}
</aether-route>
```

---

## 🔧 Python API Usage

You can also use the adapter directly in Python:

```python
from body.adapters.playcanvas_adapter import PLAYCANVAS_ADAPTER
import json
import asyncio

async def main():
    # List projects
    result = await PLAYCANVAS_ADAPTER.list_projects()
    projects = json.loads(result)
    print(projects)
    
    # Create project
    intent = json.dumps({
        "action": "create_project",
        "params": {
            "name": "My Game",
            "description": "Test game"
        }
    })
    result = await PLAYCANVAS_ADAPTER.execute(intent)
    data = json.loads(result)
    project_id = data["project_id"]
    
    # Upload script
    with open("my_script.js", "r") as f:
        content = f.read()
    
    intent = json.dumps({
        "action": "upload_script",
        "params": {
            "project_id": project_id,
            "branch_id": "master",
            "name": "myScript.js",
            "content": content
        }
    })
    result = await PLAYCANVAS_ADAPTER.execute(intent)
    print(result)

asyncio.run(main())
```

---

## 🎬 Complete Automation Example

**Goal:** Fully automate creating a PlayCanvas game connected to AetherMind

```bash
#!/bin/bash
# automated_game_setup.sh

# 1. Create project
echo "Creating PlayCanvas project..."
PROJECT_ID=$(python tools/playcanvas_cli.py create-project "Auto City" | grep "ID:" | awk '{print $2}')

# 2. Deploy AetherBridge
echo "Deploying AetherMind bridge..."
python tools/playcanvas_cli.py deploy-aether-bridge $PROJECT_ID

# 3. Upload custom game scripts
echo "Uploading game logic..."
python tools/playcanvas_cli.py upload-script $PROJECT_ID ./vehicle_controller.js
python tools/playcanvas_cli.py upload-script $PROJECT_ID ./citizen_ai.js
python tools/playcanvas_cli.py upload-script $PROJECT_ID ./camera_manager.js

# 4. Start AetherMind backend
echo "Starting AetherMind backend..."
./start_backend.sh &

# 5. Open project in browser
echo "Opening PlayCanvas Editor..."
open "https://playcanvas.com/project/$PROJECT_ID/overview"

echo "✅ Setup complete! Project ID: $PROJECT_ID"
```

---

## 🔥 Advanced: ToolForge Integration

**AetherMind can now write game code AND deploy it to PlayCanvas!**

Example flow:

1. **User:** "Create a script that makes cars drive in circles"

2. **AetherMind (ToolForge):** Generates JavaScript code:
```javascript
var CircleDrive = pc.createScript('circleDrive');
CircleDrive.attributes.add('radius', { type: 'number', default: 10 });
CircleDrive.prototype.update = function(dt) {
    const angle = Date.now() / 1000;
    const x = Math.cos(angle) * this.radius;
    const z = Math.sin(angle) * this.radius;
    this.entity.setPosition(x, 0, z);
};
```

3. **AetherMind (PlayCanvas Adapter):** Uploads to project:
```xml
<aether-route body="playcanvas">
{
  "action": "upload_script",
  "params": {
    "project_id": 789012,
    "name": "circleDrive.js",
    "content": "... [generated code] ..."
  }
}
</aether-route>
```

4. **Result:** Script immediately available in PlayCanvas Editor!

---

## 📡 API Endpoints Reference

| Action | Params | Returns |
|--------|--------|---------|
| `create_project` | name, description | project_id, url |
| `list_projects` | - | List of projects |
| `get_project` | project_id | Project details |
| `upload_script` | project_id, name, content | asset_id |
| `create_asset` | project_id, type, content | asset_id |
| `list_scenes` | project_id, branch_id | List of scenes |
| `download_app` | project_id, scenes | download_url |

---

## 🎯 Use Cases

### 1. **Demo Generation**
User signs up → AetherMind creates personalized PlayCanvas demo → Deploys AetherBridge → User plays instantly

### 2. **A/B Testing**
Create multiple project versions with different scripts, compare performance

### 3. **User-Generated Content**
Users describe game mechanics → ToolForge generates code → PlayCanvas adapter deploys → Live in seconds

### 4. **CI/CD Pipeline**
```yaml
# .github/workflows/deploy-game.yml
- name: Deploy to PlayCanvas
  run: |
    python tools/playcanvas_cli.py upload-script $PROJECT_ID ./dist/game.js
```

### 5. **Multi-Tenant Games**
Each user gets their own PlayCanvas project with custom AetherMind configurations

---

## 🔒 Security Notes

- **API Token:** Keep secret, don't commit to git
- **Rate Limits:** PlayCanvas API has rate limits (see docs)
- **HTTPS Only:** All API calls use HTTPS
- **Token Revocation:** Revoke tokens from PlayCanvas account page

---

## 🐛 Troubleshooting

### "API token not configured"
```bash
# Check .env file
cat .env | grep PLAYCANVAS_API_TOKEN

# Should show: PLAYCANVAS_API_TOKEN=xyz...
# If empty, add your token
```

### "401 Unauthorized"
Token is invalid or expired. Generate new one from playcanvas.com/account

### "429 Too many requests"
Hit rate limit. Wait 60 seconds and try again.

### "404 Project not found"
Check project ID is correct:
```bash
python tools/playcanvas_cli.py list-projects
```

---

## 📚 Resources

- **PlayCanvas API Docs:** [developer.playcanvas.com/user-manual/api](https://developer.playcanvas.com/user-manual/api/)
- **Account Settings:** [playcanvas.com/account](https://playcanvas.com/account)
- **Example Projects:** [playcanvas.com/explore](https://playcanvas.com/explore)

---

## 🎉 Next Steps

1. ✅ Get API token
2. ✅ Add to .env
3. ✅ Test CLI commands
4. 🎮 Create your first auto-generated game!
5. 🧠 Let AetherMind build games for you!

**The future is here:** AI that writes games AND deploys them! 🚀
