"""
AetherMind DCLA - Universal Body Component
Path: orchestrator/router.py
Role: Routes brain intents to the correct body adapter.
"""

from body.adapters.chat_ui import ChatAdapter
from loguru import logger

class Router:
    def __init__(self):
        """
        Initializes the router and registers the available body adapters.
        In a more complex system, this could be done dynamically.
        """
        self.adapters = {
            "chat": ChatAdapter()
        }
        logger.info("Router initialized with ChatAdapter.")

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

