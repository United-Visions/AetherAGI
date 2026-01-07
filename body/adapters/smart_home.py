"""
AetherMind DCLA - Universal Body Component
Path: body/adapters/smart_home.py

SmartHomeAdapter: Controls IoT devices as practice for physical embodiment.
Demonstrates the same Brain operating through a different Body interface.
"""

import json
from ..adapter_base import BodyAdapter
from loguru import logger


class SmartHomeAdapter(BodyAdapter):
    """
    Smart Home Body: Translates brain intents into IoT device commands.
    
    This adapter demonstrates that the same Brain can control different
    physical domains simply by switching the Body adapter.
    
    Simulated devices for demo:
    - lights (living_room, bedroom, kitchen)
    - thermostat
    - locks (front_door, back_door)
    - cameras (doorbell, backyard)
    """
    
    def __init__(self):
        # Simulated device states
        self.devices = {
            "lights": {
                "living_room": {"on": False, "brightness": 0, "color": "white"},
                "bedroom": {"on": False, "brightness": 0, "color": "white"},
                "kitchen": {"on": True, "brightness": 75, "color": "warm"}
            },
            "thermostat": {
                "current_temp": 72,
                "target_temp": 70,
                "mode": "cooling",
                "fan": "auto"
            },
            "locks": {
                "front_door": {"locked": True},
                "back_door": {"locked": True}
            },
            "cameras": {
                "doorbell": {"recording": True, "motion_detected": False},
                "backyard": {"recording": True, "motion_detected": True}
            }
        }
        logger.info("🏠 SmartHomeAdapter initialized with simulated devices")
    
    def execute(self, intent: str) -> str:
        """
        Process brain intent and execute smart home commands.
        
        Intent format (JSON):
        {
            "action": "status" | "control" | "query",
            "device_type": "lights" | "thermostat" | "locks" | "cameras",
            "device_id": "living_room" | "front_door" | etc,
            "command": { ... device-specific parameters ... }
        }
        """
        logger.info(f"🏠 SmartHomeAdapter executing intent: {intent[:100]}...")
        
        try:
            cmd = json.loads(intent)
            action = cmd.get("action", "status")
            device_type = cmd.get("device_type")
            device_id = cmd.get("device_id")
            command = cmd.get("command", {})
            
            if action == "status":
                return self._get_status(device_type, device_id)
            elif action == "control":
                return self._control_device(device_type, device_id, command)
            elif action == "query":
                return self._query_home()
            else:
                return self._format_response("error", f"Unknown action: {action}")
                
        except json.JSONDecodeError:
            return self._natural_language_response(intent)
    
    def _get_status(self, device_type: str = None, device_id: str = None) -> str:
        if device_type and device_id:
            device = self.devices.get(device_type, {}).get(device_id)
            if device:
                return self._format_response("success", {
                    "device_type": device_type,
                    "device_id": device_id,
                    "state": device
                })
            return self._format_response("error", f"Device not found: {device_type}/{device_id}")
        return self._format_response("success", {"all_devices": self.devices})
    
    def _control_device(self, device_type: str, device_id: str, command: dict) -> str:
        if device_type not in self.devices:
            return self._format_response("error", f"Unknown device type: {device_type}")
        if device_id not in self.devices[device_type]:
            return self._format_response("error", f"Unknown device: {device_id}")
        
        device = self.devices[device_type][device_id]
        changes = []
        for key, value in command.items():
            if key in device:
                old_val = device[key]
                device[key] = value
                changes.append(f"{key}: {old_val} → {value}")
                logger.info(f"🏠 [{device_type}/{device_id}] {key}: {old_val} → {value}")
        
        return self._format_response("success", {
            "device": f"{device_type}/{device_id}",
            "changes": changes,
            "new_state": device
        })
    
    def _query_home(self) -> str:
        lights_on = sum(1 for room in self.devices["lights"].values() if room["on"])
        locked = all(lock["locked"] for lock in self.devices["locks"].values())
        motion = any(cam["motion_detected"] for cam in self.devices["cameras"].values())
        
        return self._format_response("success", {
            "summary": {
                "lights_on": f"{lights_on}/{len(self.devices['lights'])}",
                "all_doors_locked": locked,
                "motion_detected": motion,
                "temperature": f"{self.devices['thermostat']['current_temp']}°F"
            }
        })
    
    def _natural_language_response(self, intent: str) -> str:
        status = self._query_home()
        status_dict = json.loads(status)
        summary = status_dict['data']['summary']
        
        return json.dumps({
            "body": "smart_home",
            "response": f"🏠 Home Status: {summary['lights_on']} lights on, "
                       f"{'locked' if summary['all_doors_locked'] else 'UNLOCKED'}, "
                       f"{summary['temperature']}",
            "intent_received": intent[:50]
        }, indent=2)
    
    def _format_response(self, status: str, data) -> str:
        return json.dumps({"status": status, "body": "smart_home", "data": data}, indent=2)
