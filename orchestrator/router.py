"""
AetherMind DCLA - Universal Body Component
Path: orchestrator/router.py
Role: Routes brain intents to the correct body adapter.
"""

from body.adapters.chat_ui import ChatAdapter
from body.adapters.practice_adapter import PracticeAdapter
from body.adapters.smart_home import SmartHomeAdapter
from body.adapters.automotive import AutomotiveAdapter
import os, json, subprocess
from body.adapters.toolforge_adapter import ToolForgeAdapter
from config.settings import settings
from loguru import logger

class Router:
    def __init__(self):
        """
        Initializes the router and registers the available body adapters.
        In a more complex system, this could be done dynamically.
        """
        self.adapters = {
            "chat": ChatAdapter(),
            "smart_home": SmartHomeAdapter(),
            "automotive": AutomotiveAdapter()
        }
        if settings.practice_adapter:
            self.adapters["practice"] = PracticeAdapter()
        if settings.toolforge_adapter:
            self.adapters["toolforge"] = ToolForgeAdapter()
        
        # Dynamic logging based on loaded adapters
        adapter_names = ", ".join(self.adapters.keys())
        logger.info(f"Router initialized with adapters: {adapter_names}")
    
    def get_available_adapters(self) -> list:
        """Return list of available adapter names."""
        return list(self.adapters.keys())

    def forward_intent(self, intent: str, adapter_type: str = "chat"):
        """
        Forwards the brain's final output (intent) to the specified adapter.
        The adapter is responsible for processing the intent into a final,
        user-facing format.
        """
        adapter = self.adapters.get(adapter_type)
        if not adapter:
            logger.error(f"No adapter found for type: {adapter_type}")
            return "ERROR: No suitable interface for this response."

        # The adapter's execute method will now return the processed output.
        return adapter.execute(intent)

