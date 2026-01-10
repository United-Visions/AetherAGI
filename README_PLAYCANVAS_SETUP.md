# 🎮 PlayCanvas + AetherMind Integration (Browser-Based!)

**NO INSTALLATION REQUIRED** - Everything runs in your browser!

PlayCanvas is a professional WebGL game engine that works entirely in the browser. Perfect for rapid prototyping and cross-platform deployment.

---

## 🚀 Quick Start (5 Minutes)

### 1. Create PlayCanvas Account

Go to **[playcanvas.com](https://playcanvas.com)** and sign up (free tier available).

### 2. Create New Project

1. Click **"New Project"**
2. Choose template (e.g., "Blank Project" or "Platformer")
3. Name it (e.g., "AetherMind Game")

### 3. Add AetherMind Bridge Script

1. In the PlayCanvas Editor, go to **Assets Panel**
2. Right-click → **New Asset → Script**
3. Name it `aetherBridge.js`
4. Open the script and paste the contents of `playcanvas_aether_bridge.js`
5. **Save** (Ctrl+S / Cmd+S)

### 4. Attach Script to Entity

1. In the **Hierarchy Panel**, select **Root** or create a new entity (Right-click → **New Entity** → Name it "GameManager")
2. In the **Inspector Panel**, click **Add Component → Script**
3. Click **Add Script** → Select `aetherBridge`
4. Configure the script attributes:
   - **API URL**: `http://localhost:8000/v1/game/unity/state` (for local testing)
   - **Sync Interval**: `500` (milliseconds)
   - **Player Entity**: Drag your player entity here
   - **Debug Mode**: ✅ Checked

### 5. Start AetherMind Backend

```bash
cd /Users/deion/Desktop/aethermind_universal
./start_backend.sh
```

### 6. Launch Your Game

1. Click the **▶️ Launch** button in PlayCanvas Editor
2. Your game opens in a new tab
3. Open browser console (F12) to see AetherMind connection logs:
   ```
   🧠 [AetherBridge] AetherMind Bridge initialized
   ✅ [AetherBridge] Connected to AetherMind backend
   ```

### 7. Test Commands via Control Panel

```bash
# In another terminal
python aether_control_panel.py
```

Click **"Move to Origin"** button → Your player teleports to (0, 0, 0)!

---

## 📡 How It Works

```
PlayCanvas Game (Browser) ←→ FastAPI Backend (localhost:8000) ←→ AetherMind Brain
         ↑                                                              ↑
    JavaScript                                                   Python Logic
```

Every 0.5 seconds:
1. PlayCanvas sends game state (player position, events, entity counts)
2. AetherMind processes state and returns commands
3. PlayCanvas executes commands (move player, spawn items, etc.)

---

## 🎮 Available Commands

### Move Entity
```javascript
{
  "action": "move",
  "target": "player",  // or entity name
  "params": {
    "x": 10,
    "y": 5,
    "z": -3
  }
}
```

### Spawn Object
```javascript
{
  "action": "spawn",
  "params": {
    "type": "enemy",  // becomes entity name and tag
    "position": { "x": 0, "y": 0, "z": 0 }
  }
}
```

### Destroy Entity
```javascript
{
  "action": "destroy",
  "target": "Enemy1"
}
```

### Set Time of Day
```javascript
{
  "action": "set_time",
  "params": {
    "hour": 12  // 0-23
  }
}
```

### Chat Message
```javascript
{
  "action": "chat",
  "params": {
    "message": "Watch out! Enemies approaching from the north!"
  }
}
```

### Switch Camera
```javascript
{
  "action": "camera_switch",
  "params": {
    "camera": "TopDownCamera"
  }
}
```

### Set Properties
```javascript
{
  "action": "set_property",
  "target": "Door",
  "params": {
    "enabled": true,
    "scale": 2.0,
    "color": { "r": 1, "g": 0, "b": 0 }
  }
}
```

---

## 💡 Custom Game Logic Examples

### Example 1: Log Event When Player Collects Item

Create a new script `collectible.js`:

```javascript
var Collectible = pc.createScript('collectible');

Collectible.prototype.initialize = function() {
    this.entity.collision.on('collisionstart', this.onCollision, this);
};

Collectible.prototype.onCollision = function(result) {
    if (result.other.name === 'Player') {
        // Get AetherMind bridge
        const bridge = this.app.root.findByName('GameManager').script.aetherBridge;
        
        // Log event to AetherMind
        bridge.logEvent(`Player collected ${this.entity.name}`);
        
        // Destroy collectible
        this.entity.destroy();
    }
};
```

Attach to any collectible entity!

### Example 2: Ask AetherMind for Advice

```javascript
var Enemy = pc.createScript('enemy');

Enemy.prototype.update = function(dt) {
    const playerDistance = this.getDistanceToPlayer();
    
    if (playerDistance < 5 && !this.askedForAdvice) {
        const bridge = this.app.root.findByName('GameManager').script.aetherBridge;
        bridge.logEvent(`Enemy ${this.entity.name} is ${playerDistance.toFixed(1)}m from player. Should I attack?`);
        this.askedForAdvice = true;
    }
};
```

AetherMind can respond with commands like:
```json
{"action": "move", "target": "Enemy1", "params": {"x": 100, "y": 0, "z": 100}}
```

### Example 3: Display AetherMind Chat in Game UI

```javascript
var UIManager = pc.createScript('uiManager');

UIManager.prototype.initialize = function() {
    // Listen for AetherMind messages
    this.app.on('aether:chat', this.onAetherMessage, this);
};

UIManager.prototype.onAetherMessage = function(message) {
    // Display in your UI (example with Element component)
    const textElement = this.entity.element;
    textElement.text = `AetherMind: ${message}`;
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        textElement.text = '';
    }, 5000);
};
```

---

## 🌐 Publishing Your Game

### Local Testing (Already Working!)
```
http://localhost:8000 (Backend)
https://playcanvas.com/editor/... (Game Editor)
https://launch.playcanvas.com/... (Running Game)
```

### Production Deployment

1. **Deploy AetherMind Backend**:
   - Use Render.com, Railway.app, or AWS
   - Get your production URL (e.g., `https://aethermind.onrender.com`)

2. **Update PlayCanvas Script**:
   - Change `apiUrl` to your production URL
   - Add authentication if needed

3. **Publish PlayCanvas Game**:
   - Click **"Publish"** in PlayCanvas Editor
   - Get shareable URL: `https://playcanv.as/p/XXXXXXX/`
   - Embed in your website or share directly

---

## 🎨 PlayCanvas Features You Get for Free

- ✅ **Real-time collaboration** - Multiple devs in same project
- ✅ **Version control** - Built-in checkpoints
- ✅ **Asset store** - Free 3D models, sounds, scripts
- ✅ **Physics engine** - Ammo.js integration
- ✅ **Mobile support** - Touch controls work automatically
- ✅ **WebGL optimization** - 60 FPS on most devices
- ✅ **No download/install** - Run on any device with a browser

---

## 🆚 PlayCanvas vs Unity

| Feature | PlayCanvas | Unity |
|---------|-----------|-------|
| **Editor** | ✅ Browser-based | ❌ Desktop app (Mac/Windows) |
| **Installation** | ✅ None | ❌ 5+ GB download |
| **Scripting** | JavaScript/TypeScript | C# |
| **3D Power** | Good (WebGL) | Excellent (Native) |
| **Load Time** | ⚡ Fast (WebAssembly) | Slower (larger builds) |
| **Cross-Platform** | ✅ Instant (any browser) | ✅ But needs separate builds |
| **AetherMind Integration** | ✅ Easy (fetch API) | ✅ UnityWebRequest |

**Recommendation**: Use **PlayCanvas** for:
- Rapid prototyping
- Browser-based games
- Mobile-first projects
- When you don't want to install Unity

Use **Unity** for:
- Complex 3D games
- VR/AR projects
- Console/Steam releases
- Advanced physics simulations

---

## 🔧 Troubleshooting

### "Failed to connect to AetherMind"

**Check:**
1. Backend is running: `curl http://localhost:8000/v1/body/list`
2. CORS enabled in `main_api.py` (already done!)
3. API URL in PlayCanvas script is correct

**Fix CORS if needed:**
```python
# In main_api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### "Entity not found" errors

Make sure your entity names match what's in the commands:
```javascript
// In PlayCanvas, rename entities to match commands
player.name = "Player";  // Exact match for target: "player"
```

### Game runs slow in browser

1. Reduce sync interval: `syncInterval: 1000` (1 second)
2. Optimize assets (compress textures)
3. Use PlayCanvas Profiler (Tools → Profiler)

---

## 🎓 Learning Resources

- **PlayCanvas Docs**: [developer.playcanvas.com/tutorials](https://developer.playcanvas.com/tutorials)
- **Examples**: [playcanvas.github.io](https://playcanvas.github.io)
- **Forum**: [forum.playcanvas.com](https://forum.playcanvas.com)
- **Discord**: PlayCanvas Community Server

---

## 🚀 Next Steps

1. ✅ **Basic Setup** - Follow Quick Start above
2. 🎮 **Add Game Logic** - Create collectibles, enemies, objectives
3. 🧠 **Let AetherMind Control NPCs** - Log events and execute commands
4. 📷 **Add Camera System** - Switch between views based on AetherMind decisions
5. 💬 **Chat Integration** - Display AetherMind messages in-game
6. 🌐 **Publish** - Share your game with the world!

---

**Have fun building!** 🎉

Questions? Check the main AetherMind docs or ask in the community.
