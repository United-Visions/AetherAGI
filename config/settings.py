"""
Path: config/settings.py
Role: Configuration loader.
"""
import os
import yaml
from loguru import logger

class Settings:
    def __init__(self):
        self._config = {}
        self.load_settings()

    def load_settings(self):
        settings_path = "config/settings.yaml"
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Failed to load settings from {settings_path}: {e}")
        else:
            # If not exists, we can default or warn.
            # Step 4 creates it, so it should exist eventually.
            pass

    def __getattr__(self, name):
        return self._config.get(name, None)

settings = Settings()
