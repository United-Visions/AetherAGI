"""
Path: perception/transcriber.py
Role: Handles audio transcription using the official OpenAI Whisper model.
"""
import whisper
from loguru import logger
import numpy as np
import io

class Transcriber:
    def __init__(self, model_name: str = "base"):
        """
        Initializes the Whisper model for audio transcription.
        Args:
            model_name (str): The name of the Whisper model to use (e.g., "base", "small", "medium").
        """
        logger.info(f"Loading Whisper model: {model_name}")
        try:
            self.model = whisper.load_model(model_name)
            logger.success("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None

    def transcribe_audio_from_bytes(self, audio_bytes: bytes) -> str:
        """
        Transcribes audio from raw bytes.
        Args:
            audio_bytes (bytes): The audio data in bytes.
        Returns:
            str: The transcribed text.
        """
        if not self.model:
            logger.error("Whisper model is not available.")
            return ""
            
        try:
            # Convert bytes to a file-like object
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav" # Whisper needs a file extension hint

            # The official library can handle the conversion directly
            audio_np = whisper.load_audio(audio_file)
            
            logger.info(f"Transcribing audio buffer...")
            result = self.model.transcribe(audio_np, fp16=False) # fp16=False for CPU compatibility
            
            text = result.get('text', '').strip()
            logger.success("Transcription completed successfully.")
            return text
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return ""

