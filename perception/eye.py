"""
Path: perception/eye.py
Role: Unified ingestion of video, images, PDFs, and audio.
"""
import av
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pdfminer.high_level import extract_text
from loguru import logger
import pytesseract

from perception.transcriber import Transcriber

class Eye:
    def __init__(self, vlm_model_name: str = "Salesforce/blip-image-captioning-large"):
        """
        Initializes the Eye with a Vision-Language Model (VLM) for image captioning/OCR
        and the Transcriber for audio.
        """
        logger.info(f"Loading Vision-Language Model: {vlm_model_name}")
        try:
            self.vlm_processor = BlipProcessor.from_pretrained(vlm_model_name)
            self.vlm_model = BlipForConditionalGeneration.from_pretrained(vlm_model_name)
            logger.success("VLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            self.vlm_processor = None
            self.vlm_model = None
            
        self.transcriber = Transcriber(model_name="base")

    def _caption_image(self, image: Image.Image) -> str:
        """Helper to caption a single PIL image."""
        if not self.vlm_model or not self.vlm_processor:
            return ""
        inputs = self.vlm_processor(images=image, return_tensors="pt")
        outputs = self.vlm_model.generate(**inputs, max_length=50)
        caption = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()

    def _ocr_image(self, image: Image.Image) -> str:
        """Helper to perform OCR on a single PIL image."""
        try:
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"Pytesseract OCR failed: {e}")
            return ""

    async def ingest(self, file_bytes: bytes, mime_type: str) -> str:
        """
        Processes a file (image, video, audio, pdf) and returns a textual representation.
        """
        logger.info(f"Eye received file with MIME type: {mime_type}")
        
        try:
            if mime_type.startswith("image"):
                image = Image.open(io.BytesIO(file_bytes))
                caption = self._caption_image(image)
                ocr_text = self._ocr_image(image)
                return f"[Image: {caption} || OCR text: {ocr_text}]"
            
            elif mime_type.startswith("audio"):
                transcript = self.transcriber.transcribe_audio_from_bytes(file_bytes)
                return f"[Audio Transcript: {transcript}]"

            elif mime_type.startswith("video"):
                captions = []
                container = av.open(io.BytesIO(file_bytes))
                audio_stream = container.streams.audio[0]
                
                # Extract and transcribe audio
                audio_frames = b''.join(p.to_bytes() for p in container.decode(audio_stream))
                transcript = self.transcriber.transcribe_audio_from_bytes(audio_frames)

                # Sample frames for captioning
                for i, frame in enumerate(container.decode(video=0)):
                    if i % 24 == 0: # Roughly one frame per second
                        captions.append(self._caption_image(frame.to_image()))
                
                summary = '; '.join(filter(None, captions))
                return f"[Video Summary: {summary} || Audio: {transcript}]"

            elif mime_type == "application/pdf":
                text = extract_text(io.BytesIO(file_bytes))
                return f"[PDF Content: {text.strip()}]"
            
            else:
                logger.warning(f"Unsupported MIME type for ingestion: {mime_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Eye ingestion failed for {mime_type}: {e}")
            return ""

