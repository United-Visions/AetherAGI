# 🎮 AetherMind Game Engine Integration

Complete setup guide for connecting games to AetherMind's cognitive architecture.

**Choose Your Engine:**
- 🌐 **[PlayCanvas](#-playcanvas-browser-based-no-install)** - Browser-based, no installation (5 min setup)
- 🎮 **[Unity](#-unity-desktop-engine)** - Professional desktop engine (requires install)

---

## 🌐 PlayCanvas (Browser-Based, No Install!)

**Best for:** Rapid prototyping, web games, mobile-first projects

### Quick Start
1. Go to **[playcanvas.com](https://playcanvas.com)** (free tier)
2. Create new project
3. Add script: `playcanvas_aether_bridge.js` (in repo root)
4. Attach to entity, configure API URL
5. Launch and play!

**Full guide:** See [README_PLAYCANVAS_SETUP.md](README_PLAYCANVAS_SETUP.md)

---

## 🎮 Unity (Desktop Engine)

**Best for:** Complex 3D games, VR/AR, console releases

## 📦 What You Get

- **Unity Adapter** - Body adapter in AetherMind that processes Unity commands
- **FastAPI Endpoints** - `/v1/game/unity/state` and `/v1/game/unity/command`
- **PySide6 Control Panel** - Cross-platform desktop app to monitor/control Unity
- **C# Bridge Script** - Ready-to-use Unity script for AetherMind connection

---

## 🚀 Quick Start

### 1. Start AetherMind Backend

```bash
cd /Users/deion/Desktop/aethermind_universal
source .venv/bin/activate
./start_backend.sh
```

Backend runs on `http://localhost:8000`

### 2. Launch Control Panel

```bash
python aether_control_panel.py
```

Features:
- ✅ Real-time Unity game state monitoring
- ✅ Send commands to Unity (move player, spawn objects, etc.)
- ✅ Chat with AetherMind Brain about game events
- ✅ Cross-platform (Mac, Windows, Linux)

### 3. Unity Integration (C# Script)

Create `Assets/Scripts/AetherBridge.cs`:

```csharp
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;
using System.Text;

[System.Serializable]
public class GameState {
    public Vector3 playerPos;
    public string currentScene;
    public List<string> events;
    public int enemyCount;
}

[System.Serializable]
public class AetherCommand {
    public string action;
    public string target;
    public Dictionary<string, object> parameters;
}

[System.Serializable]
public class AetherResponse {
    public string status;
    public List<AetherCommand> commands;
    public string aether_status;
}

public class AetherBridge : MonoBehaviour {
    [Header("AetherMind Connection")]
    public string apiUrl = "http://localhost:8000/v1/game/unity/state";
    public float syncInterval = 0.5f; // Poll every 0.5 seconds
    
    [Header("Game References")]
    public Transform playerTransform;
    public string sceneName = "MainScene";
    
    private List<string> eventQueue = new List<string>();
    private List<AetherCommand> pendingCommands = new List<AetherCommand>();
    
    void Start() {
        StartCoroutine(SyncLoop());
        Debug.Log("🧠 AetherMind Bridge Initialized");
    }
    
    IEnumerator SyncLoop() {
        while (true) {
            yield return StartCoroutine(SyncWithAether());
            yield return new WaitForSeconds(syncInterval);
        }
    }
    
    IEnumerator SyncWithAether() {
        // 1. Gather current game state
        GameState state = new GameState {
            playerPos = playerTransform ? playerTransform.position : Vector3.zero,
            currentScene = sceneName,
            events = new List<string>(eventQueue),
            enemyCount = GameObject.FindGameObjectsWithTag("Enemy").Length
        };
        
        string jsonState = JsonUtility.ToJson(state);
        
        // 2. Send to AetherMind backend
        using (UnityWebRequest request = new UnityWebRequest(apiUrl, "POST")) {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonState);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success) {
                // 3. Parse AetherMind's commands
                string responseText = request.downloadHandler.text;
                AetherResponse response = JsonUtility.FromJson<AetherResponse>(responseText);
                
                if (response.commands != null && response.commands.Count > 0) {
                    Debug.Log($"🧠 Received {response.commands.Count} command(s) from AetherMind");
                    pendingCommands.AddRange(response.commands);
                }
            } else {
                Debug.LogWarning($"⚠️ AetherMind connection failed: {request.error}");
            }
        }
        
        // 4. Execute pending commands
        ExecuteCommands();
        
        // 5. Clear processed events
        eventQueue.Clear();
    }
    
    void ExecuteCommands() {
        foreach (var cmd in pendingCommands) {
            Debug.Log($"⚡ Executing: {cmd.action} on {cmd.target}");
            
            switch (cmd.action) {
                case "move":
                    if (cmd.target == "player" && playerTransform != null) {
                        Vector3 targetPos = new Vector3(
                            GetParam<float>(cmd.parameters, "x"),
                            GetParam<float>(cmd.parameters, "y"),
                            GetParam<float>(cmd.parameters, "z")
                        );
                        playerTransform.position = targetPos;
                    }
                    break;
                    
                case "spawn":
                    string itemType = GetParam<string>(cmd.parameters, "type");
                    SpawnItem(itemType);
                    break;
                    
                case "set_time":
                    int hour = GetParam<int>(cmd.parameters, "hour");
                    SetTimeOfDay(hour);
                    break;
                    
                case "chat":
                    string message = GetParam<string>(cmd.parameters, "message");
                    ShowChatMessage(message);
                    break;
                    
                default:
                    Debug.LogWarning($"Unknown action: {cmd.action}");
                    break;
            }
        }
        
        pendingCommands.Clear();
    }
    
    // Helper to safely get parameters
    T GetParam<T>(Dictionary<string, object> dict, string key) {
        if (dict != null && dict.ContainsKey(key)) {
            return (T)dict[key];
        }
        return default(T);
    }
    
    // Game-specific implementations
    void SpawnItem(string itemType) {
        Debug.Log($"Spawning {itemType}");
        // Your spawn logic here
    }
    
    void SetTimeOfDay(int hour) {
        Debug.Log($"Setting time to {hour}:00");
        // Your lighting/time logic here
    }
    
    void ShowChatMessage(string message) {
        Debug.Log($"💬 AetherMind says: {message}");
        // Your UI logic here
    }
    
    // Public method to log events that AetherMind should know about
    public void LogEvent(string eventDescription) {
        eventQueue.Add(eventDescription);
        Debug.Log($"📝 Event logged: {eventDescription}");
    }
    
    // Example: Call this when player enters a zone
    void OnTriggerEnter(Collider other) {
        if (other.CompareTag("DangerZone")) {
            LogEvent("Player entered danger zone");
        }
    }
}
```

### 4. Unity Setup Steps

1. **Create the Bridge Script**:
   - Assets → Create → C# Script → Name it `AetherBridge.cs`
   - Copy the code above

2. **Attach to GameObject**:
   - Create an empty GameObject (Right-click in Hierarchy → Create Empty)
   - Name it "AetherMindBridge"
   - Drag `AetherBridge.cs` onto it

3. **Configure Inspector**:
   - Drag your Player GameObject into the "Player Transform" field
   - Set API URL: `http://localhost:8000/v1/game/unity/state`
   - Adjust Sync Interval (default 0.5 seconds is good)

4. **Test the Connection**:
   - Press Play in Unity
   - Open Control Panel (`python aether_control_panel.py`)
   - You should see "🟢 Connected" in the Control Panel
   - Try clicking "Move to Origin" - your player should teleport!

---

## 🎯 Use Cases

### Example 1: AI-Controlled NPC
```csharp
// In your NPC script:
void Update() {
    if (ShouldAskAether()) {
        string question = "What should I do? Enemy nearby: " + enemyDistance;
        aetherBridge.LogEvent(question);
    }
}
```

### Example 2: Dynamic Difficulty
AetherMind can analyze player performance and adjust difficulty:
```csharp
aetherBridge.LogEvent($"Player died {deathCount} times in 5 minutes");
// AetherMind might respond with: {"action": "set_difficulty", "params": {"level": "easy"}}
```

### Example 3: Procedural Events
```csharp
// AetherMind decides when to spawn events
if (response.commands.Contains("spawn_event")) {
    TriggerRandomEvent();
}
```

---

## 📡 API Reference

### Unity → AetherMind (POST /v1/game/unity/state)

**Request Body:**
```json
{
  "playerPos": {"x": 10.5, "y": 0, "z": 5.2},
  "currentScene": "Level1",
  "events": ["Player collected coin", "Enemy spawned"],
  "enemyCount": 3
}
```

**Response:**
```json
{
  "status": "synced",
  "commands": [
    {
      "action": "move",
      "target": "player",
      "params": {"x": 0, "y": 0, "z": 0}
    }
  ],
  "aether_status": "listening"
}
```

### Control Panel → AetherMind (POST /v1/game/unity/command)

**Request Body:**
```json
{
  "action": "spawn",
  "target": "item",
  "params": {"type": "health_potion", "quantity": 5}
}
```

**Response:**
```json
{
  "status": "queued",
  "queue_length": 1,
  "command": { /* echoed command */ }
}
```

---

## 🔧 Troubleshooting

### "Connection Failed" in Control Panel
1. Check backend is running: `http://localhost:8000/v1/body/list`
2. Try changing API URL to `http://127.0.0.1:8000`
3. Check firewall isn't blocking port 8000

### Unity Not Receiving Commands
1. Verify Unity console shows "🧠 AetherMind Bridge Initialized"
2. Check for errors in Unity console
3. Make sure `AetherBridge` script is active in the scene

### Commands Not Executing
1. Check `ExecuteCommands()` has the action type you're trying
2. Add debug logs: `Debug.Log($"Executing {cmd.action}")` 
3. Verify JSON parameter names match exactly

---

## 🚀 Production Deployment

### For Remote Server (e.g., Render.com)

**Unity Script Changes:**
```csharp
public string apiUrl = "https://your-aethermind-backend.onrender.com/v1/game/unity/state";
```

**Control Panel:**
Change API URL field to your production URL.

**Security:**
Add authentication:
```csharp
request.SetRequestHeader("X-Aether-Key", "your_api_key_here");
```

---

## 💡 Advanced Features

### Camera Feed Integration
```csharp
// Send screenshot to AetherMind Vision
byte[] screenshotBytes = CaptureScreenshot();
string base64Image = System.Convert.ToBase64String(screenshotBytes);
// POST to /v1/ingest/multimodal
```

### Real-time Voice Commands
```csharp
// Record audio, send to AetherMind
// Get voice-to-text response
// Execute commands
```

### Multi-Agent Systems
Run multiple Unity instances, each controlled by different AetherMind personas.

---

## 📚 Next Steps

1. ✅ **Basic Integration** - Follow this guide
2. 🎯 **Custom Commands** - Add game-specific actions to `ExecuteCommands()`
3. 🧠 **AI Behaviors** - Let AetherMind control NPCs
4. 📷 **Vision Integration** - Send camera feeds for visual analysis
5. 🌐 **Multiplayer** - Each player gets their own AetherMind instance

---

**Questions?** Check the main AetherMind docs or Discord community.
