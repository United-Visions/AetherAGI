"""
Path: perception/transcriber.py
Role: Handles audio transcription using a Whisper model.
"""
from whisper_cpp_python import Whisper
from loguru import logger
import numpy as np

class Transcriber:
    def __init__(self, model_name: str = "base"):
        """
        Initializes the Whisper model for audio transcription.
        Args:
            model_name (str): The name of the Whisper model to use (e.g., "base", "small", "medium").
        """
        logger.info(f"Loading Whisper model: {model_name}")
        try:
            self.model = Whisper(model_path=model_name)
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
            # The model expects a numpy array of f32le format
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            logger.info(f"Transcribing audio buffer of length {len(audio_np)}...")
            result = self.model.transcribe(audio_np)
            
            text = result.get('text', '').strip()
            logger.success("Transcription completed successfully.")
            return text
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return ""

