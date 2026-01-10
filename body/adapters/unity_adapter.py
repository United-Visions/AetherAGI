"""
Path: body/adapters/unity_adapter.py
Role: AetherMind Body Adapter for Unity Game Engine Integration.
      Allows the Brain to control game objects (Cars, Characters, Cameras, City Systems).
"""

import json
from body.adapter_base import BodyAdapter
from loguru import logger

class UnityAdapter(BodyAdapter):
    """
    Adapter for communicating with a Unity Game Engine instance.
    Standardizes 'intent' into Unity-compatible JSON commands.
    """
    
    def __init__(self):
        self.name = "unity_game_bridge"
        # Command queue for the game loop to poll
        self.command_queue = []
        logger.info("🎮 Unity Adapter initialized")

    def execute(self, intent: str) -> str:
        """
        Process an intent from the Brain and format it for Unity.
        Intent can be:
        1. Action JSON: {"action": "move", "target": "player", "params": {...}}
        2. Natural Language: "Open the door" (requires parsing)
        """
        try:
            # Try parsing as JSON first
            if intent.startswith("{"):
                command = json.loads(intent)
            else:
                # If text, we wrap it as a 'speak' or 'chat' command for now
                # In a real DCLA, this would go through ActionParser first
                command = {
                    "action": "chat",
                    "target": "player",
                    "params": {"message": intent}
                }
            
            # Queue the command for Unity to pick up
            self.command_queue.append(command)
            logger.info(f"🎮 Queued Unity Command: {command}")
            
            return json.dumps({
                "status": "queued",
                "queue_length": len(self.command_queue),
                "command": command
            })
            
        except json.JSONDecodeError:
            logger.error(f"❌ Failed to parse Unity intent: {intent}")
            return json.dumps({"status": "error", "message": "Invalid JSON intent"})
        except Exception as e:
            logger.error(f"❌ Unity Adapter Error: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    def get_pending_commands(self) -> list:
        """Retrieve and clear pending commands (Called by Unity polling endpoint)"""
        cmds = self.command_queue[:]
        self.command_queue = [] # Clear after reading
        return cmds

# Global instance for the API to access
UNITY_ADAPTER = UnityAdapter()
