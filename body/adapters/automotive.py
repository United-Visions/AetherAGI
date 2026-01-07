"""
AetherMind DCLA - Universal Body Component
Path: body/adapters/automotive.py

AutomotiveAdapter: Controls vehicle systems.
Demonstrates the same Brain operating through a car interface.
"""

import json
from ..adapter_base import BodyAdapter
from loguru import logger


class AutomotiveAdapter(BodyAdapter):
    """
    Automotive Body: Translates brain intents into vehicle commands.
    
    This adapter demonstrates that the same Brain can control a vehicle
    simply by switching the Body adapter.
    
    Simulated vehicle systems:
    - engine (start/stop, RPM)
    - climate (AC, heat)
    - lights (headlights, interior)
    - locks (doors, trunk)
    - navigation (destination, ETA)
    - sensors (speed, fuel, battery)
    """
    
    def __init__(self):
        # Simulated vehicle state
        self.vehicle = {
            "engine": {
                "running": False,
                "rpm": 0,
                "gear": "P"
            },
            "climate": {
                "ac_on": False,
                "temperature": 72,
                "fan_speed": 0
            },
            "lights": {
                "headlights": "off",  # off, auto, on, high
                "interior": False,
                "turn_signals": "off"  # off, left, right, hazard
            },
            "locks": {
                "doors": "locked",
                "trunk": "locked",
                "windows": "closed"
            },
            "navigation": {
                "destination": None,
                "eta": None,
                "distance_remaining": None
            },
            "sensors": {
                "speed_mph": 0,
                "fuel_percent": 75,
                "battery_percent": 100,
                "tire_pressure": {"fl": 32, "fr": 32, "rl": 31, "rr": 31}
            }
        }
        logger.info("🚗 AutomotiveAdapter initialized with simulated vehicle")
    
    def execute(self, intent: str) -> str:
        """
        Process brain intent and execute vehicle commands.
        
        Intent format (JSON):
        {
            "action": "status" | "control" | "navigate" | "query",
            "system": "engine" | "climate" | "lights" | "locks" | "navigation",
            "command": { ... system-specific parameters ... }
        }
        """
        logger.info(f"🚗 AutomotiveAdapter executing intent: {intent[:100]}...")
        
        try:
            cmd = json.loads(intent)
            action = cmd.get("action", "status")
            system = cmd.get("system")
            command = cmd.get("command", {})
            
            if action == "status":
                return self._get_status(system)
            elif action == "control":
                return self._control_system(system, command)
            elif action == "navigate":
                return self._set_navigation(command)
            elif action == "query":
                return self._query_vehicle()
            else:
                return self._format_response("error", f"Unknown action: {action}")
                
        except json.JSONDecodeError:
            return self._natural_language_response(intent)
    
    def _get_status(self, system: str = None) -> str:
        if system and system in self.vehicle:
            return self._format_response("success", {
                "system": system,
                "state": self.vehicle[system]
            })
        return self._format_response("success", {"vehicle": self.vehicle})
    
    def _control_system(self, system: str, command: dict) -> str:
        if system not in self.vehicle:
            return self._format_response("error", f"Unknown system: {system}")
        
        changes = []
        for key, value in command.items():
            if key in self.vehicle[system]:
                old_val = self.vehicle[system][key]
                self.vehicle[system][key] = value
                changes.append(f"{key}: {old_val} → {value}")
                logger.info(f"🚗 [{system}] {key}: {old_val} → {value}")
        
        # Special handling for engine start
        if system == "engine" and command.get("running"):
            self.vehicle["engine"]["rpm"] = 800  # Idle
            changes.append("Engine started - idle at 800 RPM")
        
        return self._format_response("success", {
            "system": system,
            "changes": changes,
            "new_state": self.vehicle[system]
        })
    
    def _set_navigation(self, command: dict) -> str:
        destination = command.get("destination")
        if destination:
            self.vehicle["navigation"]["destination"] = destination
            self.vehicle["navigation"]["eta"] = "25 min"
            self.vehicle["navigation"]["distance_remaining"] = "12.5 miles"
            logger.info(f"🚗 Navigation set to: {destination}")
        
        return self._format_response("success", {
            "navigation": self.vehicle["navigation"]
        })
    
    def _query_vehicle(self) -> str:
        engine = self.vehicle["engine"]
        sensors = self.vehicle["sensors"]
        
        return self._format_response("success", {
            "summary": {
                "engine": "Running" if engine["running"] else "Off",
                "gear": engine["gear"],
                "speed": f"{sensors['speed_mph']} mph",
                "fuel": f"{sensors['fuel_percent']}%",
                "doors": self.vehicle["locks"]["doors"]
            }
        })
    
    def _natural_language_response(self, intent: str) -> str:
        status = self._query_vehicle()
        status_dict = json.loads(status)
        summary = status_dict['data']['summary']
        
        return json.dumps({
            "body": "automotive",
            "response": f"🚗 Vehicle Status: Engine {summary['engine']}, "
                       f"Gear: {summary['gear']}, Speed: {summary['speed']}, "
                       f"Fuel: {summary['fuel']}, Doors: {summary['doors']}",
            "intent_received": intent[:50]
        }, indent=2)
    
    def _format_response(self, status: str, data) -> str:
        return json.dumps({"status": status, "body": "automotive", "data": data}, indent=2)
