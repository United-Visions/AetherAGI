# perception_service/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from loguru import logger
import sys

# Add parent directory to path to import Eye and Transcriber
sys.path.append("..") 
from perception.eye import Eye
from perception.transcriber import Transcriber

# --- Configuration & Model Loading ---
# This section runs only ONCE when the worker starts.
logger.info("Starting Perception Service worker...")
try:
    # We instantiate the Eye here. It will load the VLM and Whisper models into memory.
    # This can take a few minutes the first time the worker spins up.
    eye_service = Eye()
    logger.success("Eye service and all models loaded successfully.")
except Exception as e:
    logger.error(f"FATAL: Could not load models. Worker will not function. Error: {e}")
    eye_service = None

# --- FastAPI App ---
# This is the standard web server application.
app = FastAPI(title="AetherMind Perception Service", version="1.0")

@app.post("/v1/ingest")
async def ingest_multimodal_data(file: UploadFile = File(...)):
    """
    Accepts a file upload and uses the Eye service to process it.
    """
    if not eye_service:
        raise HTTPException(status_code=500, detail="Perception models are not available.")

    logger.info(f"Received file for ingestion: {file.filename} ({file.content_type})")
    
    try:
        file_bytes = await file.read()
        mime_type = file.content_type
        
        # Use the pre-loaded Eye instance to process the data
        text_representation = await eye_service.ingest(file_bytes=file_bytes, mime_type=mime_type)
        
        if not text_representation:
            raise HTTPException(status_code=400, detail="Could not extract any content from the file.")
            
        logger.success("File ingestion successful.")
        return {
            "status": "success",
            "filename": file.filename,
            "mime_type": mime_type,
            "text_representation": text_representation
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed for file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok" if eye_service else "degraded"}
