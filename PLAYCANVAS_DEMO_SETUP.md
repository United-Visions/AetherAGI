# 🏙️ AetherMind PlayCanvas Demo - Complete Setup Guide

**Goal:** Showcase AetherMind controlling a virtual city with all body adapters working together.

---

## 🎯 Demo Concept: "The Connected City"

A simple 3D city where AetherMind controls:
- 🚗 Smart vehicles (automotive adapter)
- 🏠 Smart buildings (smart_home adapter)
- 👤 NPCs/Citizens (chat adapter)
- 📷 Surveillance cameras (vision_system adapter)
- 🎮 Game mechanics (unity adapter)
- 🔧 Dynamic tools (toolforge adapter)

---

## 📦 Free Assets You Need (All 100% Free!)

### 1. **City Buildings** - Quaternius Low Poly City Pack
🔗 [quaternius.com](https://quaternius.com/packs/ultimatetexturedbuildings.html)
- 50+ modular buildings
- Street props (benches, trees, lights)
- CC0 License (use anywhere, no attribution)

**How to import to PlayCanvas:**
1. Download `.fbx` or `.obj` files
2. In PlayCanvas Editor: Right-click Assets → Upload
3. Drag into scene

### 2. **Cars & Vehicles** - Kenney Car Kit
🔗 [kenney.nl/assets/car-kit](https://kenney.nl/assets/car-kit)
- Low-poly cars, trucks, buses
- CC0 License

### 3. **People/NPCs** - Mixamo Characters
🔗 [mixamo.com](https://www.mixamo.com/)
- Free animated characters
- Walk, run, idle animations included

### 4. **Camera Assets** - Built into PlayCanvas
No download needed! PlayCanvas includes:
- Orbit camera script
- First-person controller
- Multi-camera switcher

---

## 🚀 Step-by-Step Setup (30 Minutes)

### Phase 1: Create Base City (10 min)

1. **Go to [playcanvas.com/editor](https://playcanvas.com/editor)**
2. **Create New Project** → Name: "AetherMind Connected City"
3. **Download free assets**:
   - Quaternius Buildings: [Ultimate Modular Buildings](https://quaternius.com)
   - Kenney Cars: [Car Kit](https://kenney.nl/assets/car-kit)
   - Or use built-in primitives (cubes for buildings, capsules for people)

4. **Build simple city**:
   ```
   Scene hierarchy:
   ├── Ground (flat plane, 100x100)
   ├── Buildings (5-10 building models)
   ├── Roads (stretched cubes with dark material)
   ├── Cars (3-5 vehicles)
   ├── Citizens (5-10 capsules with walk animations)
   ├── Cameras
   │   ├── OrbitCamera (God view)
   │   ├── StreetCamera (street-level)
   │   └── SurveillanceCamera (fixed security cam)
   ├── Lights
   │   ├── DirectionalLight (Sun)
   │   └── PointLights (street lamps)
   └── GameManager (holds AetherBridge script)
   ```

### Phase 2: Add AetherMind Bridge (5 min)

1. **Create Script**:
   - Assets → New Asset → Script
   - Name: `aetherBridge.js`
   - Copy from `/playcanvas_aether_bridge.js`

2. **Attach to GameManager**:
   - Create empty entity: "GameManager"
   - Add Script component
   - Select `aetherBridge`
   - Configure:
     - API URL: `http://localhost:8000/v1/game/unity/state`
     - Player Entity: Drag your main character
     - Debug Mode: ✅ Checked

### Phase 3: Add Camera Switching (5 min)

Create `cameraManager.js`:

```javascript
var CameraManager = pc.createScript('cameraManager');

CameraManager.attributes.add('cameras', {
    type: 'entity',
    array: true,
    title: 'Camera List'
});

CameraManager.prototype.initialize = function() {
    this.currentIndex = 0;
    this.switchCamera(0);
    
    // Listen for AetherMind commands
    this.app.on('aether:camera', this.onAetherSwitch, this);
};

CameraManager.prototype.switchCamera = function(index) {
    // Disable all cameras
    this.cameras.forEach(cam => {
        if (cam.camera) cam.camera.enabled = false;
    });
    
    // Enable target camera
    if (this.cameras[index] && this.cameras[index].camera) {
        this.cameras[index].camera.enabled = true;
        this.currentIndex = index;
        console.log(`📷 Switched to: ${this.cameras[index].name}`);
    }
};

CameraManager.prototype.onAetherSwitch = function(cameraName) {
    const index = this.cameras.findIndex(c => c.name === cameraName);
    if (index >= 0) this.switchCamera(index);
};

// Keyboard shortcut for testing
CameraManager.prototype.update = function(dt) {
    if (this.app.keyboard.wasPressed(pc.KEY_C)) {
        this.switchCamera((this.currentIndex + 1) % this.cameras.length);
    }
};
```

### Phase 4: Add Vehicle Movement (5 min)

Create `vehicle.js`:

```javascript
var Vehicle = pc.createScript('vehicle');

Vehicle.attributes.add('speed', {
    type: 'number',
    default: 2
});

Vehicle.attributes.add('waypoints', {
    type: 'entity',
    array: true
});

Vehicle.prototype.initialize = function() {
    this.currentWaypoint = 0;
    this.moving = true;
};

Vehicle.prototype.update = function(dt) {
    if (!this.moving || !this.waypoints.length) return;
    
    const target = this.waypoints[this.currentWaypoint].getPosition();
    const pos = this.entity.getPosition();
    const direction = target.clone().sub(pos).normalize();
    
    // Move towards waypoint
    pos.add(direction.scale(this.speed * dt));
    this.entity.setPosition(pos);
    
    // Look in movement direction
    this.entity.lookAt(target);
    
    // Check if reached waypoint
    if (pos.distance(target) < 0.5) {
        this.currentWaypoint = (this.currentWaypoint + 1) % this.waypoints.length;
    }
};
```

### Phase 5: Add NPC Behaviors (5 min)

Create `citizen.js`:

```javascript
var Citizen = pc.createScript('citizen');

Citizen.attributes.add('walkSpeed', {
    type: 'number',
    default: 1.5
});

Citizen.prototype.initialize = function() {
    this.state = 'idle'; // idle, walking, talking
    this.idleTime = 0;
    this.targetPos = null;
};

Citizen.prototype.update = function(dt) {
    switch (this.state) {
        case 'idle':
            this.idleTime += dt;
            if (this.idleTime > 3) {
                this.startWalking();
            }
            break;
            
        case 'walking':
            if (this.targetPos) {
                const pos = this.entity.getPosition();
                const dir = this.targetPos.clone().sub(pos).normalize();
                pos.add(dir.scale(this.walkSpeed * dt));
                this.entity.setPosition(pos);
                
                if (pos.distance(this.targetPos) < 0.5) {
                    this.state = 'idle';
                    this.idleTime = 0;
                }
            }
            break;
    }
};

Citizen.prototype.startWalking = function() {
    this.state = 'walking';
    this.targetPos = new pc.Vec3(
        Math.random() * 20 - 10,
        0,
        Math.random() * 20 - 10
    );
};
```

---

## 🧠 AetherMind Commands for Demo

Once setup is complete, use the control panel to showcase:

### 1. **Surveillance Mode** (Vision Adapter)
```json
{
  "action": "camera_switch",
  "params": {"camera": "SurveillanceCamera"}
}
```

### 2. **Control Traffic** (Automotive Adapter)
```json
{
  "action": "set_property",
  "target": "Car1",
  "params": {"enabled": false}
}
```

### 3. **Move Citizen** (Chat/Unity Adapter)
```json
{
  "action": "move",
  "target": "Citizen3",
  "params": {"x": 10, "y": 0, "z": 5}
}
```

### 4. **Time Control** (Smart Home Adapter - Lighting)
```json
{
  "action": "set_time",
  "params": {"hour": 0}
}
```

### 5. **Spawn Emergency** (ToolForge Adapter)
```json
{
  "action": "spawn",
  "params": {"type": "emergency_vehicle", "position": {"x": 0, "y": 0, "z": 0}}
}
```

---

## 🎬 Demo Script (Show All Body Types)

**Opening (30 seconds):**
1. Launch PlayCanvas game
2. Show city overview from OrbitCamera
3. Point out: vehicles moving, citizens walking, lights on

**AetherMind Takes Control (2 minutes):**

```bash
# Terminal 1: Backend
./start_backend.sh

# Terminal 2: Control Panel
python aether_control_panel.py
```

**Live Demo:**
1. **Chat Adapter** - Type in control panel: "What's happening in the city?"
   - AetherMind reads game state, responds about vehicle count, citizen positions
   
2. **Vision Adapter** - Click "Switch to Surveillance Camera"
   - Camera changes to security view
   
3. **Automotive Adapter** - Send command: Stop Car 2
   - Vehicle freezes in place
   
4. **Smart Home Adapter** - Click "Set Night"
   - Lights dim, streetlights turn on
   
5. **Unity Adapter** - Custom command: Spawn collectible
   - New item appears in scene
   
6. **ToolForge Adapter** - (if time) Create custom tool on the fly
   - Show code generation in sandbox

**Closing (10 seconds):**
- Show all systems working together
- "AetherMind - One brain, every interface"

---

## 📊 What This Demonstrates

| AetherMind Body | Demo Showcase |
|-----------------|---------------|
| **Chat** | Natural language queries about game state |
| **Vision** | Camera switching and surveillance |
| **Automotive** | Vehicle control and traffic management |
| **Smart Home** | Environmental controls (lighting, time) |
| **Unity/Game** | Spawn objects, move entities, game mechanics |
| **ToolForge** | Runtime tool creation (advanced) |
| **Hardware** | (Optional) Connect real GPIO/Serial devices |

---

## 💾 Save & Share

Once built:

1. **Save Project** - Auto-saves in PlayCanvas
2. **Publish** - Click "Publish" button → Get shareable URL
3. **Embed** - Copy iframe code for your website
4. **Present** - Share link: `https://playcanv.as/p/XXXXX/`

---

## 🎨 Styling Options (All Free)

### Minimal/Low-Poly Style
- ✅ Fast loading
- ✅ Works on mobile
- ✅ Clean, professional look

### Realistic Style
- Download Pixar RenderMan assets
- Add PBR materials from [FreePBR.com](https://freepbr.com)
- Enable shadows and lighting effects

### Cyberpunk/Futuristic
- Neon colors on buildings
- Emissive materials
- Particle effects (built into PlayCanvas)

---

## ⚡ Pro Tips

1. **Tag Everything**:
   ```javascript
   car.tags.add('vehicle');
   building.tags.add('structure');
   citizen.tags.add('npc');
   ```
   Makes AetherMind commands easier!

2. **Use Prefabs**:
   Create one car/citizen, make it a prefab, duplicate 10x

3. **LOD (Level of Detail)**:
   PlayCanvas auto-optimizes, but keep poly count <10k per model

4. **Mobile Testing**:
   Click Launch → Opens on phone too!

---

## 🚀 Ready to Build?

**Total time:** ~30 minutes for basic demo  
**Total cost:** $0 (all assets and tools are free)  
**Result:** Professional demo showcasing AetherMind's full capabilities

**Next steps:**
1. Sign up: [playcanvas.com](https://playcanvas.com)
2. Download assets: [quaternius.com](https://quaternius.com), [kenney.nl](https://kenney.nl)
3. Follow setup guide above
4. Test with control panel!

---

**Questions?** Check PlayCanvas tutorials: [developer.playcanvas.com/tutorials](https://developer.playcanvas.com/tutorials)
