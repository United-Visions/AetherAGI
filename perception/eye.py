"""
Path: perception/eye.py
Role: Unified ingestion of video, images, PDFs, and audio using LiteLLM (Google Gemini) and Whisper.
"""
import av
import io
import base64
from PIL import Image
from pdfminer.high_level import extract_text
from loguru import logger
import litellm

from perception.transcriber import Transcriber

class Eye:
    def __init__(self, vision_model: str = "gemini/gemini-2.5-pro"):
        """
        Initializes the Eye with a Vision-Language Model (VLM) via LiteLLM
        and the Transcriber for audio.
        """
        self.vision_model = vision_model
        logger.info(f"Eye initialized with Vision Model: {vision_model}")
        
        # Initialize Transcriber (uses local Whisper or API)
        self.transcriber = Transcriber(model_name="whisper-1")

    async def _analyze_image(self, image: Image.Image, prompt: str = "Describe this image in detail. Transcribe any text visible.") -> str:
        """Helper to analyze a single PIL image using LiteLLM."""
        try:
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            response = await litellm.acompletion(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ]
                    }
                ],
                max_tokens=2048  # Increased for detailed visual analysis
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return "[Error analyzing image]"

    async def ingest(self, file_bytes: bytes, mime_type: str) -> str:
        """
        Processes a file (image, video, audio, pdf) and returns a textual representation.
        """
        logger.info(f"Eye received file with MIME type: {mime_type}")
        
        try:
            if mime_type.startswith("image"):
                image = Image.open(io.BytesIO(file_bytes))
                analysis = await self._analyze_image(image)
                return f"[Image Analysis: {analysis}]"
            
            elif mime_type.startswith("audio"):
                transcript = await self.transcriber.transcribe_audio_from_bytes(file_bytes)
                return f"[Audio Transcript: {transcript}]"

            elif mime_type.startswith("video"):
                captions = []
                container = av.open(io.BytesIO(file_bytes))
                
                # 1. Extract and transcribe audio
                try:
                    audio_stream = container.streams.audio[0]
                    audio_frames = b''.join(p.to_bytes() for p in container.decode(audio_stream))
                    transcript = await self.transcriber.transcribe_audio_from_bytes(audio_frames)
                except Exception as e:
                    logger.warning(f"Could not extract audio from video: {e}")
                    transcript = "[No audio track]"

                # 2. Sample frames for visual analysis (e.g., 1 frame every 2 seconds)
                # For efficiency with API costs, we'll take fewer frames than local
                frames = []
                for i, frame in enumerate(container.decode(video=0)):
                    if i % 48 == 0: # Approx every 2 seconds (assuming 24fps)
                        frames.append(frame.to_image())
                
                # Analyze key frames (limit to 5 to save tokens/latency)
                selected_frames = frames[:5] 
                for i, frame in enumerate(selected_frames):
                    caption = await self._analyze_image(frame, prompt=f"Describe this frame from a video (Frame {i+1}).")
                    captions.append(f"Frame {i+1}: {caption}")
                
                summary = '\n'.join(captions)
                return f"[Video Analysis]\nVisuals:\n{summary}\n\nAudio:\n{transcript}"

            elif mime_type == "application/pdf":
                text = extract_text(io.BytesIO(file_bytes))
                return f"[PDF Content: {text.strip()}]"
            
            else:
                logger.warning(f"Unsupported MIME type for ingestion: {mime_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Eye ingestion failed for {mime_type}: {e}")
            return f"Error processing file: {str(e)}"


