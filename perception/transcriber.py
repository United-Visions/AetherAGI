"""
Path: perception/transcriber.py
Role: Handles audio transcription using LiteLLM (supporting OpenAI Whisper, Groq, etc.).
"""
import io
import os
from loguru import logger
import litellm

class Transcriber:
    def __init__(self, model_name: str = "whisper-1"):
        """
        Initializes the Transcriber with an API-based model.
        Args:
            model_name (str): The name of the model to use (e.g., "whisper-1", "groq/whisper-large-v3").
        """
        self.model_name = model_name
        logger.info(f"Initialized Transcriber with model: {self.model_name}")

    async def transcribe_audio_from_bytes(self, audio_bytes: bytes) -> str:
        """
        Transcribes audio from raw bytes using LiteLLM.
        Args:
            audio_bytes (bytes): The audio data in bytes.
        Returns:
            str: The transcribed text.
        """
        try:
            # LiteLLM/OpenAI API expects a file-like object with a name
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.mp3"  # Defaulting to mp3, but wav works too

            logger.info(f"Sending audio to {self.model_name} for transcription...")
            
            # litellm.transcription is synchronous, so we wrap it if needed, 
            # but for now we call it directly.
            response = litellm.transcription(
                model=self.model_name,
                file=audio_file
            )
            
            text = response.text.strip()
            logger.success(f"Transcription completed: {text[:50]}...")
            return text
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return ""

