"""
3D Embodiment Adapter - Real-time control of the 3D doll visualization

Bridges the gap between Aether's brain and the Three.js 3D game environment.
Receives action tags from the brain and sends real-time commands to the frontend
via WebSocket.

Action Tags Supported:
- <aether-3d-move target="x,y,z" speed="walk|run">Navigate doll to position</aether-3d-move>
- <aether-3d-look target="x,y,z|object_name">Turn doll to face direction</aether-3d-look>
- <aether-3d-animation name="idle|walk|run|wave|dance">Play animation</aether-3d-animation>
- <aether-3d-explore mode="start|stop" radius="10">Start/stop exploration</aether-3d-explore>
- <aether-3d-teleport x="0" y="0" z="0">Instant position change</aether-3d-teleport>
- <aether-3d-emotion type="happy|sad|surprised|curious">Display emotion</aether-3d-emotion>
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from body.adapter_base import BodyAdapter

class Embodiment3DAdapter(BodyAdapter):
    """
    Adapter for controlling the 3D doll in the game environment.
    Receives commands from the brain and broadcasts to connected WebSocket clients.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "embodiment_3d"
        self.description = "Control the 3D character visualization in real-time"
        
        # WebSocket connections to game clients
        self.websocket_clients: List[Any] = []
        
        # Current doll state
        self.doll_state = {
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0},
            "animation": "idle",
            "emotion": "neutral",
            "is_moving": False,
            "is_exploring": False,
            "current_goal": None,
            "last_update": datetime.now().isoformat()
        }
        
        # Perception buffer from game
        self.perception_buffer = {
            "body_state": None,
            "vision": None,
            "spatial_memory": None,
            "last_received": None
        }
        
        print("🎮 3D Embodiment Adapter initialized")
    
    async def execute(self, intent: str) -> str:
        """
        Execute a 3D embodiment command.
        
        Args:
            intent: JSON string with action details
            
        Returns:
            Execution result message
        """
        try:
            spec = json.loads(intent)
            action_type = spec.get("action_type", "")
            
            if action_type == "move":
                return await self._handle_move(spec)
            elif action_type == "look":
                return await self._handle_look(spec)
            elif action_type == "animation":
                return await self._handle_animation(spec)
            elif action_type == "explore":
                return await self._handle_explore(spec)
            elif action_type == "teleport":
                return await self._handle_teleport(spec)
            elif action_type == "emotion":
                return await self._handle_emotion(spec)
            elif action_type == "vision":
                return await self._handle_vision(spec)
            elif action_type == "get_state":
                return await self._get_current_state()
            elif action_type == "get_perception":
                return await self._get_perception()
            else:
                return f"Unknown 3D action type: {action_type}"
                
        except json.JSONDecodeError as e:
            return f"Invalid JSON in 3D embodiment intent: {e}"
        except Exception as e:
            return f"Error executing 3D embodiment action: {e}"
    
    async def _handle_move(self, spec: Dict[str, Any]) -> str:
        """Move doll to target position"""
        target = spec.get("target", {})
        speed = spec.get("speed", "walk")
        
        # Update state
        self.doll_state["is_moving"] = True
        self.doll_state["current_goal"] = f"Moving to {target}"
        
        # Send command to all connected clients
        command = {
            "type": "move",
            "target": target,
            "speed": speed,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_command(command)
        
        return f"✓ Moving to position ({target.get('x', 0)}, {target.get('y', 0)}, {target.get('z', 0)}) at {speed} speed"
    
    async def _handle_look(self, spec: Dict[str, Any]) -> str:
        """Turn doll to face target"""
        target = spec.get("target", {})
        
        command = {
            "type": "look",
            "target": target,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_command(command)
        
        return f"✓ Turning to face {target}"
    
    async def _handle_animation(self, spec: Dict[str, Any]) -> str:
        """Play animation on doll"""
        animation_name = spec.get("name", "idle")
        loop = spec.get("loop", True)
        
        self.doll_state["animation"] = animation_name
        
        command = {
            "type": "animation",
            "name": animation_name,
            "loop": loop,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_command(command)
        
        return f"✓ Playing animation: {animation_name}"
    
    async def _handle_explore(self, spec: Dict[str, Any]) -> str:
        """Start/stop exploration mode"""
        mode = spec.get("mode", "start")
        radius = spec.get("radius", 10)
        
        self.doll_state["is_exploring"] = (mode == "start")
        
        command = {
            "type": "explore",
            "mode": mode,
            "radius": radius,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_command(command)
        
        return f"✓ Exploration mode: {mode} (radius: {radius}m)"
    
    async def _handle_teleport(self, spec: Dict[str, Any]) -> str:
        """Instantly move doll to position"""
        x = spec.get("x", 0)
        y = spec.get("y", 0)
        z = spec.get("z", 0)
        
        self.doll_state["position"] = {"x": x, "y": y, "z": z}
        
        command = {
            "type": "teleport",
            "position": {"x": x, "y": y, "z": z},
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_command(command)
        
        return f"✓ Teleported to ({x}, {y}, {z})"
    
    async def _handle_emotion(self, spec: Dict[str, Any]) -> str:
        """Display emotion on doll"""
        emotion_type = spec.get("type", "neutral")
        duration = spec.get("duration", 3000)  # milliseconds
        
        self.doll_state["emotion"] = emotion_type
        
        command = {
            "type": "emotion",
            "emotion": emotion_type,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_command(command)
        
        return f"✓ Expressing emotion: {emotion_type}"
    
    async def _handle_vision(self, spec: Dict[str, Any]) -> str:
        """Control camera/gaze direction for active vision"""
        action = spec.get("action", "look")
        direction = spec.get("direction", "front")
        degrees = spec.get("degrees", 45)
        angles = spec.get("angles", "left,front,right")
        target = spec.get("target")
        steps = spec.get("steps", 8)
        
        command = {
            "type": "vision",
            "action": action,
            "direction": direction,
            "degrees": degrees,
            "angles": angles,
            "target": target,
            "steps": steps,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_command(command)
        
        # Build descriptive result
        if action == "look":
            return f"✓ Looking {direction} ({degrees}°)"
        elif action == "scan":
            return f"✓ Scanning area: {angles}"
        elif action == "focus":
            return f"✓ Focusing on target point"
        elif action == "panorama":
            return f"✓ Capturing {steps}-angle panorama"
        elif action == "peripheral":
            return f"✓ Capturing peripheral vision"
        elif action == "vertical":
            return f"✓ Performing vertical scan"
        elif action == "reset":
            return f"✓ Camera reset to original view"
        else:
            return f"✓ Vision command executed: {action}"
    
    async def _get_current_state(self) -> str:
        """Get current doll state"""
        return json.dumps({
            "success": True,
            "state": self.doll_state
        }, indent=2)
    
    async def _get_perception(self) -> str:
        """Get latest perception data from game"""
        if not self.perception_buffer.get("last_received"):
            return json.dumps({
                "success": False,
                "message": "No perception data received yet"
            })
        
        return json.dumps({
            "success": True,
            "perception": self.perception_buffer
        }, indent=2)
    
    async def _broadcast_command(self, command: Dict[str, Any]):
        """Send command to all connected WebSocket clients"""
        if not self.websocket_clients:
            print(f"⚠️ No WebSocket clients connected - command queued: {command['type']}")
            return
        
        # Wrap command in proper format: {type, data, timestamp}
        wrapped_command = {
            "type": command.pop("type"),
            "data": command,  # All other fields go in data
            "timestamp": command.pop("timestamp", datetime.now().isoformat())
        }
        message = json.dumps(wrapped_command)
        
        # Send to all clients (remove disconnected ones)
        disconnected = []
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except Exception as e:
                print(f"⚠️ Client disconnected: {e}")
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.websocket_clients.remove(client)
    
    def register_websocket(self, websocket):
        """Register a new WebSocket client"""
        self.websocket_clients.append(websocket)
        print(f"🔌 WebSocket client connected (total: {len(self.websocket_clients)})")
    
    def unregister_websocket(self, websocket):
        """Remove a WebSocket client"""
        if websocket in self.websocket_clients:
            self.websocket_clients.remove(websocket)
            print(f"🔌 WebSocket client disconnected (remaining: {len(self.websocket_clients)})")
    
    def update_perception(self, perception_data: Dict[str, Any]):
        """
        Update perception buffer with data from game.
        Called by the API endpoint when game sends perception updates.
        """
        self.perception_buffer.update({
            "body_state": perception_data.get("body_state"),
            "vision": perception_data.get("vision"),
            "spatial_memory": perception_data.get("spatial_memory"),
            "last_received": datetime.now().isoformat()
        })
        
        # Update doll state from body state
        if perception_data.get("body_state"):
            body = perception_data["body_state"]
            if body.get("position"):
                self.doll_state["position"] = body["position"]
            if body.get("rotation"):
                self.doll_state["rotation"] = body["rotation"]
            self.doll_state["is_moving"] = body.get("isMoving", False)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return adapter capabilities for Brain discovery"""
        return {
            "name": self.name,
            "description": self.description,
            "supported_actions": [
                "move", "look", "animation", "explore", 
                "teleport", "emotion", "get_state", "get_perception"
            ],
            "real_time": True,
            "requires_websocket": True,
            "connected_clients": len(self.websocket_clients)
        }


# Singleton instance
_embodiment_adapter_instance = None

def get_embodiment_adapter() -> Embodiment3DAdapter:
    """Get or create the singleton embodiment adapter"""
    global _embodiment_adapter_instance
    if _embodiment_adapter_instance is None:
        _embodiment_adapter_instance = Embodiment3DAdapter()
    return _embodiment_adapter_instance
