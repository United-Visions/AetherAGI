"""
AetherMind DCLA - Universal Body Component
Path: body/adapters/chat_ui.py
"""

from .adapter_base import BodyAdapter

class ChatAdapter(BodyAdapter):
    def execute(self, intent):
        print(f'Sending to Chat UI: {intent}')