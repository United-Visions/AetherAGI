# PlayCanvas + AetherMind Manual Setup Guide

## Overview
This guide walks you through manually integrating AetherMind into your PlayCanvas project. No automation required - just copy, paste, and configure!

---

## Prerequisites

1. **PlayCanvas Account**: Create one at [playcanvas.com](https://playcanvas.com) (free tier works)
2. **AetherMind Backend**: Running locally or deployed
   ```bash
   # Start backend (if local)
   ./start_backend.sh
   # Backend runs on http://localhost:8000
   ```

---

## Step 1: Create a New PlayCanvas Project

1. Go to [playcanvas.com](https://playcanvas.com) and log in
2. Click **"New Project"**
3. Choose **"Blank Project"** or any starter template
4. Name it (e.g., "AetherMind Game")
5. Click **"Create"**

---

## Step 2: Open the Editor

1. From your project dashboard, click **"Editor"** button
2. Wait for the editor to fully load (you'll see the 3D viewport)

---

## Step 3: Create the AetherBridge Script

1. In the **Assets Panel** (bottom left), right-click → **New Asset** → **Script**
2. Name it: `aetherBridge.js`
3. Double-click the new script to open the **Code Editor**
4. **Delete all existing code** and paste the following:

```javascript
/**
 * AetherMind Bridge for PlayCanvas
 * 
 * Browser-based game engine integration - No installation required!
 * Works in PlayCanvas Editor: https://playcanvas.com
 */

var AetherBridge = pc.createScript('aetherBridge');

// Script Attributes (appear in PlayCanvas Inspector)
AetherBridge.attributes.add('apiUrl', {
    type: 'string',
    default: 'http://localhost:8000/v1/game/unity/state',
    title: 'AetherMind API URL',
    description: 'Backend endpoint for game state sync'
});

AetherBridge.attributes.add('syncInterval', {
    type: 'number',
    default: 500,
    title: 'Sync Interval (ms)',
    description: 'How often to sync with AetherMind (milliseconds)'
});

AetherBridge.attributes.add('playerEntity', {
    type: 'entity',
    title: 'Player Entity',
    description: 'Reference to the player entity'
});

AetherBridge.attributes.add('debugMode', {
    type: 'boolean',
    default: true,
    title: 'Debug Mode',
    description: 'Log messages to console'
});

// Initialize
AetherBridge.prototype.initialize = function() {
    this.eventQueue = [];
    this.pendingCommands = [];
    this.isConnected = false;
    
    // Start sync loop
    this.syncTimer = setInterval(() => {
        this.syncWithAether();
    }, this.syncInterval);
    
    this.log('🧠 AetherMind Bridge initialized');
    this.log(`Connecting to: ${this.apiUrl}`);
};

// Main sync function - sends state, receives commands
AetherBridge.prototype.syncWithAether = function() {
    const gameState = this.collectGameState();
    
    fetch(this.apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Aether-Key': 'dev_mode'
        },
        body: JSON.stringify(gameState)
    })
    .then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    })
    .then(data => {
        if (!this.isConnected) {
            this.isConnected = true;
            this.log('✅ Connected to AetherMind backend', 'success');
        }
        
        if (data.commands && data.commands.length > 0) {
            this.log(`🧠 Received ${data.commands.length} command(s)`, 'info');
            data.commands.forEach(cmd => this.executeCommand(cmd));
        }
        
        this.eventQueue = [];
    })
    .catch(error => {
        if (this.isConnected) {
            this.log(`⚠️ Connection lost: ${error.message}`, 'warning');
            this.isConnected = false;
        }
    });
};

// Collect current game state
AetherBridge.prototype.collectGameState = function() {
    const state = {
        timestamp: Date.now(),
        currentScene: this.app.scene.name || 'Untitled',
        events: [...this.eventQueue],
        entities: {}
    };
    
    if (this.playerEntity) {
        const pos = this.playerEntity.getPosition();
        const rot = this.playerEntity.getRotation().getEulerAngles();
        
        state.entities.player = {
            position: { x: pos.x, y: pos.y, z: pos.z },
            rotation: { x: rot.x, y: rot.y, z: rot.z },
            active: this.playerEntity.enabled
        };
    }
    
    state.entityCounts = {
        enemies: this.app.root.findByTag('enemy').length,
        collectibles: this.app.root.findByTag('collectible').length,
        npcs: this.app.root.findByTag('npc').length
    };
    
    state.performance = {
        fps: Math.round(this.app.stats.fps),
        drawCalls: this.app.stats.drawCalls,
        triangles: this.app.stats.triangles
    };
    
    return state;
};

// Execute command from AetherMind
AetherBridge.prototype.executeCommand = function(cmd) {
    this.log(`⚡ Executing: ${cmd.action} on ${cmd.target || 'scene'}`, 'info');
    
    const params = cmd.params || {};
    
    switch (cmd.action) {
        case 'move':
            this.handleMove(cmd.target, params);
            break;
        case 'spawn':
            this.handleSpawn(params);
            break;
        case 'destroy':
            this.handleDestroy(cmd.target);
            break;
        case 'set_time':
            this.handleSetTime(params);
            break;
        case 'chat':
            this.handleChat(params);
            break;
        case 'camera_switch':
            this.handleCameraSwitch(params);
            break;
        case 'set_property':
            this.handleSetProperty(cmd.target, params);
            break;
        default:
            this.log(`⚠️ Unknown action: ${cmd.action}`, 'warning');
    }
};

// Command Handlers
AetherBridge.prototype.handleMove = function(target, params) {
    let entity = (target === 'player' && this.playerEntity) 
        ? this.playerEntity 
        : this.app.root.findByName(target);
    
    if (entity) {
        const pos = entity.getPosition();
        entity.setPosition(
            params.x !== undefined ? params.x : pos.x,
            params.y !== undefined ? params.y : pos.y,
            params.z !== undefined ? params.z : pos.z
        );
        this.log(`Moved ${target} to (${params.x}, ${params.y}, ${params.z})`);
    }
};

AetherBridge.prototype.handleSpawn = function(params) {
    const type = params.type || 'default';
    const pos = params.position || { x: 0, y: 0, z: 0 };
    
    const entity = new pc.Entity(type);
    entity.addComponent('model', { type: 'box' });
    entity.setPosition(pos.x, pos.y, pos.z);
    entity.tags.add(type);
    
    this.app.root.addChild(entity);
    this.log(`Spawned ${type} at (${pos.x}, ${pos.y}, ${pos.z})`);
};

AetherBridge.prototype.handleDestroy = function(target) {
    const entity = this.app.root.findByName(target);
    if (entity) {
        entity.destroy();
        this.log(`Destroyed ${target}`);
    }
};

AetherBridge.prototype.handleSetTime = function(params) {
    const hour = params.hour || 12;
    const lights = this.app.root.findByTag('sun');
    lights.forEach(light => {
        if (light.light) {
            light.light.intensity = (hour >= 6 && hour <= 18) ? 1.0 : 0.1;
        }
    });
    this.log(`Set time to ${hour}:00`);
};

AetherBridge.prototype.handleChat = function(params) {
    const message = params.message || params.text || 'Hello!';
    this.log(`💬 AetherMind says: ${message}`, 'success');
    this.app.fire('aether:chat', message);
};

AetherBridge.prototype.handleCameraSwitch = function(params) {
    const cameraName = params.camera || params.name;
    const camera = this.app.root.findByName(cameraName);
    
    if (camera && camera.camera) {
        this.app.root.findByTag('camera').forEach(cam => {
            if (cam.camera) cam.camera.enabled = false;
        });
        camera.camera.enabled = true;
        this.log(`Switched to camera: ${cameraName}`);
    }
};

AetherBridge.prototype.handleSetProperty = function(target, params) {
    const entity = this.app.root.findByName(target);
    if (!entity) return;
    
    if (params.enabled !== undefined) entity.enabled = params.enabled;
    if (params.scale !== undefined) entity.setLocalScale(params.scale, params.scale, params.scale);
    if (params.color !== undefined && entity.model) {
        const material = entity.model.meshInstances[0].material;
        material.diffuse.set(params.color.r, params.color.g, params.color.b);
        material.update();
    }
    this.log(`Updated properties for ${target}`);
};

// Public API
AetherBridge.prototype.logEvent = function(eventDescription) {
    this.eventQueue.push({
        timestamp: Date.now(),
        description: eventDescription
    });
    this.log(`📝 Event logged: ${eventDescription}`);
};

AetherBridge.prototype.log = function(message, level = 'info') {
    if (!this.debugMode) return;
    const prefix = { info: '🔵', success: '✅', warning: '⚠️', error: '❌' }[level] || '🔵';
    console.log(`${prefix} [AetherBridge] ${message}`);
};

AetherBridge.prototype.destroy = function() {
    if (this.syncTimer) clearInterval(this.syncTimer);
    this.log('AetherBridge destroyed');
};

AetherBridge.prototype.update = function(dt) {
    // Per-frame logic here if needed
};
```

5. Press **Ctrl+S** (or Cmd+S on Mac) to save
6. Close the code editor tab

---

## Step 4: Create a GameManager Entity

1. In the **Hierarchy Panel** (left side), right-click → **New Entity** → **Entity**
2. Rename it to `GameManager` (click on it, then edit name in Inspector)
3. With `GameManager` selected, click **"Add Component"** in the Inspector (right side)
4. Select **"Script"**
5. In the Script component, click **"Add Script"**
6. Select `aetherBridge` from the list

---

## Step 5: Configure the Bridge

With `GameManager` selected, you'll see these options in the Inspector:

| Setting | Value | Description |
|---------|-------|-------------|
| **API URL** | `http://localhost:8000/v1/game/unity/state` | Your AetherMind backend |
| **Sync Interval** | `500` | Sync every 500ms |
| **Player Entity** | (drag your player here) | Optional |
| **Debug Mode** | ✓ | See console logs |

### For Production Deployment:
Change API URL to your deployed backend:
```
https://your-aethermind-server.com/v1/game/unity/state
```

---

## Step 6: Create a Simple Player (Optional)

1. Right-click in Hierarchy → **New Entity** → **Box**
2. Rename to `Player`
3. Drag `Player` entity onto the **Player Entity** field in GameManager's script settings

---

## Step 7: Test the Connection

1. **Start your AetherMind backend** (if not running):
   ```bash
   ./start_backend.sh
   ```

2. In PlayCanvas, click the **Launch** button (play icon, top right)

3. Open **Browser DevTools** (F12) → **Console** tab

4. You should see:
   ```
   🔵 [AetherBridge] 🧠 AetherMind Bridge initialized
   🔵 [AetherBridge] Connecting to: http://localhost:8000/v1/game/unity/state
   ✅ [AetherBridge] ✅ Connected to AetherMind backend
   ```

---

## Step 8: Send Commands from AetherMind

### Via Chat Interface
Talk to AetherMind and mention your game:
> "Move the player to position 5, 0, 3"

### Via API (for testing)
```bash
curl -X POST http://localhost:8000/v1/game/command \
  -H "Content-Type: application/json" \
  -d '{
    "commands": [
      {"action": "move", "target": "player", "params": {"x": 5, "y": 0, "z": 3}}
    ]
  }'
```

---

## Available Commands

| Action | Target | Params | Example |
|--------|--------|--------|---------|
| `move` | entity name | `{x, y, z}` | Move player to position |
| `spawn` | - | `{type, position}` | Create new entity |
| `destroy` | entity name | - | Remove entity |
| `set_time` | - | `{hour}` | Change lighting |
| `chat` | - | `{message}` | Display message |
| `camera_switch` | - | `{camera}` | Switch active camera |
| `set_property` | entity name | `{enabled, scale, color}` | Modify properties |

---

## Logging Game Events

From any other script in your game, you can log events that AetherMind will receive:

```javascript
// Get the bridge
var bridge = this.app.root.findByName('GameManager').script.aetherBridge;

// Log events
bridge.logEvent('Player collected coin');
bridge.logEvent('Enemy spawned at wave 3');
bridge.logEvent('Boss defeated!');
```

---

## Listening for AetherMind Messages

```javascript
// In any script's initialize():
this.app.on('aether:chat', function(message) {
    console.log('AetherMind says:', message);
    // Display in your game UI
}, this);
```

---

## Troubleshooting

### "Connection lost" in console
- Ensure backend is running: `./start_backend.sh`
- Check API URL is correct
- Check for CORS errors in console (may need to configure backend)

### Script not appearing in "Add Script" list
- Make sure you saved the script (Ctrl+S)
- Try refreshing the editor (Ctrl+R)
- Check for syntax errors in the code editor

### No console output
- Enable Debug Mode in the script settings
- Open browser DevTools (F12) → Console

---

## Next Steps

1. **Add more game objects** and tag them (`enemy`, `collectible`, `npc`)
2. **Create a Player controller** with movement
3. **Build a UI** to display AetherMind chat messages
4. **Deploy your game** and update the API URL

---

## Quick Reference

```
PlayCanvas Editor URL: https://playcanvas.com/editor/project/YOUR_PROJECT_ID
AetherMind Backend:    http://localhost:8000 (local)
API Endpoint:          /v1/game/unity/state
```

**You're all set!** 🎮🧠
