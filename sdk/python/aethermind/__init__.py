"""
AetherMind Python SDK
Official Python client for AetherMind AGI
"""

__version__ = "1.0.0"
__author__ = "AetherMind Team"
__license__ = "Apache-2.0"

from .client import AetherMindClient, AetherMindError, AuthenticationError, RateLimitError
from .models import ChatMessage, MemoryQuery, ToolCall, UserRole

__all__ = [
    "AetherMindClient",
    "AetherMindError",
    "AuthenticationError",
    "RateLimitError",
    "ChatMessage",
    "MemoryQuery",
    "ToolCall",
    "UserRole",
]
