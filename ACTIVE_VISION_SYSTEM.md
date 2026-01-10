# Active Vision System ✅

Aether can now **actively control its vision** by moving the camera/gaze to look around the environment. This is essential for object detection and interaction.

## What Changed

### 1. **Camera Control in VisionSystem.js**
- **Look in directions**: left, right, up, down, front
- **Scan multiple angles**: panoramic capture with configurable steps
- **Focus on points**: Point camera at specific 3D coordinates
- **Multi-angle capture**: Peripheral vision (left/center/right), vertical scan (up/center/down)
- **Camera state tracking**: Remembers original view and can reset

### 2. **New Action Tag: `<aether-3d-vision>`**
The Brain can now control what Aether looks at:

```xml
<!-- Look in a direction -->
<aether-3d-vision action="look" direction="left" degrees="45">
Check what's to the left
</aether-3d-vision>

<!-- Scan area (multi-angle) -->
<aether-3d-vision action="scan" angles="left,front,right">
Scan the surroundings
</aether-3d-vision>

<!-- Focus on specific point -->
<aether-3d-vision action="focus" target="x,y,z">
{"target": {"x": 10, "y": 1, "z": 5}}
</aether-3d-vision>

<!-- Capture 360° panorama -->
<aether-3d-vision action="panorama" steps="8">
Get a complete view of the area
</aether-3d-vision>

<!-- Peripheral vision -->
<aether-3d-vision action="peripheral">
Check sides without turning body
</aether-3d-vision>

<!-- Vertical scan -->
<aether-3d-vision action="vertical">
Look up and down
</aether-3d-vision>

<!-- Reset camera -->
<aether-3d-vision action="reset">
Return to original view
</aether-3d-vision>
```

### 3. **Vision Methods Available**

#### VisionSystem Methods
```javascript
// Look in a direction
vision.lookInDirection("left", 45);  // Look left 45°
vision.lookInDirection("right", 90); // Look right 90°

// Scan multiple angles
await vision.scanArea(["left", "front", "right"]);

// Focus on 3D point
vision.focusOnPoint({x: 10, y: 1, z: 5});

// Capture panorama
await vision.capturePanorama(8);  // 8 angles = 360°/8 = 45° each

// Peripheral vision
await vision.capturePeripheralVision();

// Vertical scan
await vision.captureVerticalScan();

// Reset camera
vision.resetCamera();
```

## How It Works

### Active Vision Flow

```
1. Brain wants to find an object
   ↓
2. Generates: <aether-3d-vision action="scan" angles="left,front,right">
   ↓
3. Backend parser → Embodiment3DAdapter
   ↓
4. WebSocket broadcast to game client
   ↓
5. EmbodimentController.handleVisionCommand()
   ↓
6. VisionSystem.scanArea(["left", "front", "right"])
   ↓
7. Camera rotates to each angle and captures frame
   ↓
8. Frames sent to Eye module (Gemini Vision) for analysis
   ↓
9. Brain receives: "I can see a red cube to my left"
```

### Camera Rotation System

The VisionSystem tracks:
- **Original rotation**: Saved when first looking around
- **Yaw offset**: Left/right rotation (Y-axis)
- **Pitch offset**: Up/down rotation (X-axis)
- **Multi-angle cache**: Stores recent captures from different angles

When you call `lookInDirection("left", 45)`:
1. Saves original camera rotation (if not already saved)
2. Calculates rotation offset: `45° * π/180 = 0.785 radians`
3. Applies offset to camera: `camera.rotation.y += offset`
4. Marks `isLooking = true`

When you call `resetCamera()`:
1. Restores original rotation
2. Clears offsets
3. Marks `isLooking = false`

## Use Cases

### Finding Objects
```
User: "Can you find the red cube?"

Brain thinks:
1. <aether-3d-vision action="scan" angles="left,front,right">
   Scan for the cube
2. (Sees red cube in left view)
3. <aether-3d-vision action="focus" target="cube_position">
   Look directly at it
4. Response: "Yes, I found a red cube to my left at position (5, 0.5, 3)"
```

### Navigation
```
User: "What's ahead?"

Brain:
1. <aether-3d-vision action="look" direction="front">
   Look straight ahead
2. (Analyzes view)
3. Response: "I see a pathway with buildings on both sides"
```

### Exploration
```
User: "Look around"

Brain:
1. <aether-3d-vision action="panorama" steps="8">
   Capture 360° view
2. (Processes all 8 angles)
3. Response: "I can see: a fountain to the north, buildings to the east and west, 
   and an open plaza to the south"
```

### Object Interaction
```
User: "Pick up the object next to you"

Brain:
1. <aether-3d-vision action="peripheral">
   Check left and right sides
2. (Sees object on right)
3. <aether-3d-vision action="focus" target="object_position">
   Focus on it
4. <aether-3d-move target="object_position">
   Walk to it
5. (Future: pickup command)
```

## Architecture

### Frontend Changes
- ✅ [VisionSystem.js](aether_environment/src/perception/VisionSystem.js)
  - Added camera rotation methods
  - Added multi-angle capture
  - Added panorama capture
  - Added focus/reset methods

- ✅ [EmbodimentController.js](aether_environment/src/body/EmbodimentController.js)
  - Added `handleVisionCommand()`
  - Integrated with WebSocket command router

### Backend Changes
- ✅ [embodiment_3d_adapter.py](body/adapters/embodiment_3d_adapter.py)
  - Added `_handle_vision()` method
  - Broadcasts vision commands via WebSocket

- ✅ [system_prompts.py](brain/system_prompts.py)
  - Added full documentation for `<aether-3d-vision>` tag
  - Includes 7 action types with examples

- ✅ [action_parser.py](orchestrator/action_parser.py)
  - Added `aether-3d-vision` pattern
  - Integrated into execution pipeline

## Testing

### 1. Start the System
```bash
# Backend
./start_backend.sh

# Frontend
cd aether_environment && npm run dev
```

### 2. Test Camera Control
Open browser console and try:
```javascript
// Access the embodiment controller
const embodiment = window.embodimentController; // (if exposed)

// Look left
embodiment.vision.lookInDirection("left", 45);

// Scan area
await embodiment.vision.scanArea(["left", "front", "right"]);

// Reset
embodiment.vision.resetCamera();
```

### 3. Test via Chat
Say to Aether:
- **"What's to your left?"** → Should look left and describe
- **"Look around"** → Should scan or capture panorama
- **"Can you see the fountain?"** → Should actively search for it
- **"Look up"** → Should rotate camera upward

### 4. Monitor Commands
Watch browser console for:
```
👁️ Looking left (45°)
👁️ Scanning area: left, front, right
👁️ Area scan complete
👁️ Focusing on target
👁️ Camera reset
```

## Next Steps: Object Interaction

Now that Aether can **actively look around**, we're ready for:

### 1. **Add Interactive Objects**
```javascript
// Create clickable/pickable objects
const cube = createInteractiveObject({
  type: "cube",
  color: "red",
  position: {x: 5, y: 0.5, z: 3},
  interactive: true
});
```

### 2. **Object Detection**
When vision system captures frames:
- Send to Gemini Vision: "List all objects you see"
- Parse response: "red cube, blue sphere, fountain"
- Store object positions in spatial memory

### 3. **Object Interaction Commands**
```xml
<aether-3d-interact object="red_cube" action="pick_up">
Pick up the red cube
</aether-3d-interact>

<aether-3d-interact object="door" action="open">
Open the door
</aether-3d-interact>
```

### 4. **Task Completion**
```
User: "Move the red cube to the fountain"

Brain:
1. <aether-3d-vision action="scan"> - Find cube
2. <aether-3d-move target="cube"> - Walk to cube
3. <aether-3d-interact object="cube" action="pick_up"> - Pick it up
4. <aether-3d-vision action="scan"> - Find fountain
5. <aether-3d-move target="fountain"> - Walk to fountain
6. <aether-3d-interact object="cube" action="place"> - Place cube
7. Response: "Done! I moved the red cube to the fountain."
```

## Benefits

✅ **True Active Perception**: No longer passive - actively looks for things  
✅ **Object Finding**: Can locate objects by scanning environment  
✅ **Spatial Awareness**: Builds mental map from multi-angle captures  
✅ **Navigation Prep**: Looks ahead before moving  
✅ **Task Capability**: Foundation for "find and manipulate" tasks  
✅ **Human-like Behavior**: Turns head/gaze like a real person  
✅ **Scalable**: Easy to add more vision actions (track, follow, etc.)  

---

**Status**: 🟢 **FULLY OPERATIONAL**

Aether now has active vision control and is ready for object interaction tasks!
