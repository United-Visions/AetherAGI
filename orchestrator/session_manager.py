"""
Path: orchestrator/session_manager.py
Role: Managing User Personas and Session State.
"""

class SessionManager:
    def __init__(self):
        # In a real app, this would connect to Supabase
        self.user_sessions = {}

    def get_user_persona(self, user_id: str):
        """
        Returns the personality settings for a user.
        """
        # Default persona
        return self.user_sessions.get(user_id, {
            "name": "User",
            "tone": "helpful and logical",
            "expertise": "generalist"
        })

    def set_user_persona(self, user_id: str, settings: dict):
        self.user_sessions[user_id] = settings