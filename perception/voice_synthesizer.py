"""
Voice Synthesizer - Edge TTS Integration for AetherMind

Provides text-to-speech capabilities using Microsoft Edge's neural voices.
Free, high-quality voices without API keys.
"""

import edge_tts
import asyncio
import io
import base64
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class VoiceConfig:
    """Voice configuration for a persona"""
    voice_id: str = "en-US-AriaNeural"  # Default female voice
    rate: str = "+0%"  # Speed: -50% to +100%
    pitch: str = "+0Hz"  # Pitch adjustment
    volume: str = "+0%"  # Volume adjustment


# Predefined voice profiles for different personas
VOICE_PROFILES = {
    "default": VoiceConfig(voice_id="en-US-AriaNeural"),
    "professional": VoiceConfig(voice_id="en-US-JennyNeural", rate="-5%"),
    "casual": VoiceConfig(voice_id="en-US-SaraNeural", rate="+5%"),
    "energetic": VoiceConfig(voice_id="en-US-AnaNeural", rate="+10%", pitch="+5Hz"),
    "calm": VoiceConfig(voice_id="en-US-AriaNeural", rate="-10%", pitch="-2Hz"),
    "male_professional": VoiceConfig(voice_id="en-US-GuyNeural", rate="-5%"),
    "male_casual": VoiceConfig(voice_id="en-US-ChristopherNeural"),
    "british_female": VoiceConfig(voice_id="en-GB-SoniaNeural"),
    "british_male": VoiceConfig(voice_id="en-GB-RyanNeural"),
    "australian_female": VoiceConfig(voice_id="en-AU-NatashaNeural"),
    "australian_male": VoiceConfig(voice_id="en-AU-WilliamNeural"),
    # Persona-specific voices
    "spoiled_brat": VoiceConfig(voice_id="en-US-AnaNeural", rate="+15%", pitch="+8Hz"),
    "seo_lady": VoiceConfig(voice_id="en-US-JennyNeural", rate="+5%"),
    "karen": VoiceConfig(voice_id="en-US-MichelleNeural", rate="+10%", pitch="+3Hz"),
}


class VoiceSynthesizer:
    """
    Synthesizes speech from text using Edge TTS.
    
    Features:
    - Multiple voice profiles
    - Persona-specific voices
    - SSML support for emphasis
    - Streaming audio generation
    - Clean text preprocessing (removes markdown, think tags, etc.)
    """
    
    def __init__(self):
        self.default_voice = "en-US-AriaNeural"
        self._voices_cache: Optional[List[Dict]] = None
    
    async def list_voices(self, language_filter: str = "en") -> List[Dict[str, Any]]:
        """
        List all available voices, optionally filtered by language.
        
        Args:
            language_filter: Language code prefix (e.g., "en", "es", "fr")
        
        Returns:
            List of voice dictionaries with name, gender, locale
        """
        if self._voices_cache is None:
            voices = await edge_tts.list_voices()
            self._voices_cache = voices
        
        if language_filter:
            return [
                {
                    "voice_id": v["ShortName"],
                    "name": v["FriendlyName"],
                    "gender": v["Gender"],
                    "locale": v["Locale"],
                }
                for v in self._voices_cache
                if v["Locale"].startswith(language_filter)
            ]
        return self._voices_cache
    
    def _clean_text_for_speech(self, text: str) -> str:
        """
        Clean text for speech synthesis by removing:
        - <think> tags and their content
        - Markdown formatting
        - Code blocks
        - URLs
        - Special characters
        """
        # Remove <think> blocks entirely
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', 'code block omitted', text)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove markdown headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove markdown bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove markdown links, keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove bullet points
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        
        # Remove numbered lists formatting
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def get_voice_for_persona(self, persona_name: Optional[str] = None) -> VoiceConfig:
        """
        Get the appropriate voice configuration for a persona.
        
        Args:
            persona_name: Name of the persona (case-insensitive)
        
        Returns:
            VoiceConfig for the persona
        """
        if persona_name:
            key = persona_name.lower().replace(" ", "_")
            if key in VOICE_PROFILES:
                return VOICE_PROFILES[key]
        return VOICE_PROFILES["default"]
    
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        persona: Optional[str] = None,
        output_format: str = "audio-24khz-48kbitrate-mono-mp3"
    ) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID (overrides persona voice)
            rate: Speech rate adjustment
            pitch: Pitch adjustment  
            volume: Volume adjustment
            persona: Persona name to use voice profile from
            output_format: Audio format
        
        Returns:
            Audio bytes (MP3 by default)
        """
        # Clean the text
        clean_text = self._clean_text_for_speech(text)
        
        if not clean_text:
            return b""
        
        # Determine voice settings
        if persona and not voice_id:
            config = self.get_voice_for_persona(persona)
            voice_id = config.voice_id
            rate = config.rate
            pitch = config.pitch
            volume = config.volume
        
        voice_id = voice_id or self.default_voice
        
        # Create communicator with settings
        communicate = edge_tts.Communicate(
            text=clean_text,
            voice=voice_id,
            rate=rate,
            pitch=pitch,
            volume=volume
        )
        
        # Collect audio chunks
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        
        return b"".join(audio_chunks)
    
    async def synthesize_to_base64(
        self,
        text: str,
        **kwargs
    ) -> str:
        """
        Synthesize speech and return as base64-encoded string.
        Useful for sending audio directly to frontend.
        
        Returns:
            Base64-encoded audio string
        """
        audio_bytes = await self.synthesize(text, **kwargs)
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    async def synthesize_streaming(
        self,
        text: str,
        voice_id: Optional[str] = None,
        persona: Optional[str] = None
    ):
        """
        Generator that yields audio chunks for streaming playback.
        
        Yields:
            Audio chunk bytes
        """
        clean_text = self._clean_text_for_speech(text)
        
        if not clean_text:
            return
        
        if persona and not voice_id:
            config = self.get_voice_for_persona(persona)
            voice_id = config.voice_id
        
        voice_id = voice_id or self.default_voice
        
        communicate = edge_tts.Communicate(text=clean_text, voice=voice_id)
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]


# Singleton instance
_synthesizer: Optional[VoiceSynthesizer] = None


def get_voice_synthesizer() -> VoiceSynthesizer:
    """Get or create the voice synthesizer singleton."""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = VoiceSynthesizer()
    return _synthesizer


# Quick helper functions
async def speak(text: str, persona: Optional[str] = None) -> bytes:
    """Quick helper to synthesize speech."""
    synth = get_voice_synthesizer()
    return await synth.synthesize(text, persona=persona)


async def speak_base64(text: str, persona: Optional[str] = None) -> str:
    """Quick helper to synthesize speech as base64."""
    synth = get_voice_synthesizer()
    return await synth.synthesize_to_base64(text, persona=persona)
