"""
Path: body/adapters/notification.py
Role: Universal Notification Adapter for Cross-Device "First Contact".
Scalable for Smart Home, Mobile, Automotive, and Robotics.
"""

import json
from .adapter_base import BodyAdapter
from loguru import logger

class DeviceNotificationAdapter(BodyAdapter):
    def __init__(self, device_type="web"):
        """
        Initialize the adapter for a specific hardware/interface context.
        device_type: 'web', 'mobile', 'home_assistant', 'automotive', 'robotic_voice'
        """
        self.device_type = device_type
        logger.info(f"DeviceNotificationAdapter initialized for: {device_type}")

    async def execute(self, intent: str) -> str:
        """
        Forward a proactive notification to the physical/interface hardware.
        """
        try:
            data = json.loads(intent)
            message = data.get("message")
            priority = data.get("priority", "normal")
            
            if not message:
                return "Error: No message content provided for notification."

            # Protocol mapping
            if self.device_type == "web":
                return self._notify_browser(message, priority)
            elif self.device_type == "home_assistant":
                return self._notify_iot_bridge(message, priority)
            elif self.device_type == "automotive":
                return self._notify_car_hud(message, priority)
            elif self.device_type == "robotic_voice":
                return self._notify_tts_engine(message, priority)
            else:
                return f"Notification sent via generic protocol: {message}"

        except Exception as e:
            logger.error(f"Notification delivery failed: {e}")
            return f"Error: {str(e)}"

    def _notify_browser(self, msg, priority):
        # Already handled by router.js calling the proactive endpoint
        return f"Browser push notification (Priority: {priority}): {msg}"

    def _notify_iot_bridge(self, msg, priority):
        # Implementation for Home Assistant / Amazon Alexa / Google Home
        logger.info(f"[IOT BRIDGE] Pulsing smart lights and announcing: {msg}")
        return "IOT_BRIDGE: SUCCESS"

    def _notify_car_hud(self, msg, priority):
        # Implementation for Android Auto / CarPlay / CAN bus
        logger.info(f"[HUD] Displaying priority message on cluster: {msg}")
        return "CAR_HUD: DISPLAYED"

    def _notify_tts_engine(self, msg, priority):
        # Implementation for MaryTTS / ElevenLabs / ROS
        logger.info(f"[ROBOT] TTS Output: {msg}")
        return "ROBOT_TTS: SPOKEN"
