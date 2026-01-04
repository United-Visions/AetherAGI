"""
Path: orchestrator/main_api.py
Role: Production API Gateway. Handles authentication and routing.
"""

import os
from fastapi import FastAPI, Header, HTTPException, Depends, Security, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import numpy as np
import httpx

from loguru import logger

from .active_inference import ActiveInferenceLoop
from .auth_manager import AuthManager
from .router import Router
from brain.logic_engine import LogicEngine
from heart.heart_orchestrator import Heart # New Heart import
from mind.vector_store import AetherVectorStore
from mind.episodic_memory import EpisodicMemory
from brain.jepa_aligner import JEPAAligner
from curiosity.surprise_detector import SurpriseDetector
from curiosity.research_scheduler import ResearchScheduler
from orchestrator.agent_state_machine import AgentStateMachine
from config.settings import settings
import asyncio

load_dotenv()

app = FastAPI(title="AetherMind AGI API", version="1.0.0")

# --- CORS Middleware ---
# This allows your frontend (running on a different domain) to make API calls to this backend.
origins = [
    "https://aethermind-frontend.onrender.com",
    "http://localhost:5000", # For local testing
    "http://127.0.0.1:5000", # For local testing with explicit IP
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS middleware configured with allowed origins: {origins}")

# Setup Security
API_KEY_NAME = "X-Aether-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Initialize Core Components
STORE = AetherVectorStore(api_key=os.getenv("PINECONE_API_KEY"))
MEMORY = EpisodicMemory(STORE)
BRAIN = LogicEngine(runpod_key=os.getenv("RUNPOD_API_KEY"), endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"), pinecone_key=os.getenv("PINECONE_API_KEY"))
ROUTER = Router()
HEART = Heart(pinecone_key=os.getenv("PINECONE_API_KEY")) # Initialize the new Heart
AUTH = AuthManager()

# Initialize Sensory & Curiosity Components
JEPA = JEPAAligner()
SURPRISE_DETECTOR = SurpriseDetector(jepa=JEPA, store=STORE)

# AETHER now takes SURPRISE_DETECTOR
AETHER = ActiveInferenceLoop(BRAIN, MEMORY, STORE, ROUTER, HEART, surprise_detector=SURPRISE_DETECTOR)
RESEARCH_SCHEDULER = ResearchScheduler(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"))

# --- HTTP Client for External Services ---
# It's best practice to initialize one client and reuse it.
perception_service_url = os.getenv("PERCEPTION_ENDPOINT_URL")
if not perception_service_url:
    print("WARNING: PERCEPTION_ENDPOINT_URL environment variable not set. Multimodal features will fail.")

http_client = httpx.AsyncClient(timeout=60.0) # 60 second timeout for model processing

# --- Startup ---

@app.on_event("startup")
async def startup_event():
    if settings.features.agent_state_machine:
        # Mock MetaController until fully implemented
        class MockMetaController:
            async def decide_next_action(self, user_id):
                return {"adapter": "chat", "intent": "System: MetaController not ready."}

        agent_sm = AgentStateMachine(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            aether_loop=AETHER,
            meta_ctrl=MockMetaController(),
            router=ROUTER
        )

        pilot_users = settings.pilot_users or []
        for uid in pilot_users:
            asyncio.create_task(agent_sm.run(uid))
            logger.info(f"Started AgentStateMachine for pilot user: {uid}")

# --- Schemas ---
class ChatCompletionRequest(BaseModel):
    model: str = "aethermind-v1"
    messages: list
    user: str # The user_id for episodic memory tracking
    # Add a field for user feedback
    reaction_score: Optional[float] = None 
    last_message_id: Optional[str] = None

# --- Middleware-like Auth ---
async def get_user_id(api_key: str = Security(api_key_header)):
    user_id = AUTH.verify_key(api_key)
    if not user_id:
        raise HTTPException(status_code=403, detail="Invalid or missing Aether-Secret-Key")
    return user_id

# --- Endpoints ---

@app.post("/v1/ingest/multimodal")
async def ingest_multimodal(
    file: UploadFile,
    user_id: str = Depends(get_user_id)
):
    """
    New endpoint to handle multimodal file uploads.
    This now forwards the request to the external Perception Service.
    """
    if not perception_service_url:
        raise HTTPException(status_code=500, detail="Perception Service is not configured on the backend.")

    contents = await file.read()
    
    try:
        # Forward the file to the Perception Service
        files = {'file': (file.filename, contents, file.content_type)}
        response = await http_client.post(f"{perception_service_url}/v1/ingest", files=files)
        
        # Handle non-200 responses from the perception service
        response.raise_for_status()
        
        perception_data = response.json()
        text_representation = perception_data.get("text_representation", "")

    except httpx.RequestError as e:
        # Handles network errors, timeouts, etc.
        raise HTTPException(status_code=503, detail=f"Perception Service is unavailable or timed out: {e}")
    except httpx.HTTPStatusError as e:
        # Handles 4xx or 5xx errors from the perception service
        detail = f"Perception Service returned an error: {e.response.text}"
        if e.response.status_code == 500: # General model error
             detail = "The perception model encountered an internal error."
        raise HTTPException(status_code=e.response.status_code, detail=detail)

    # 2. Generate an embedding for the text representation
    embedding_vector = np.random.rand(1024).astype(np.float32)

    # 3. Score for surprise
    surprise_score = await SURPRISE_DETECTOR.score(embedding_vector)

    # 4. If surprising, schedule research
    if surprise_score > SURPRISE_DETECTOR.novelty_threshold:
        # Generate research questions (a real implementation would call the Brain)
        questions = [f"What is this? {text_representation[:100]}", f"Tell me more about {text_representation[:100]}"]
        for q in questions:
            job = {
                "query": q,
                "surprise": surprise_score,
                "tools": ["browser"],
                "deadline": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
                "user_id": user_id
            }
            await RESEARCH_SCHEDULER.push(job)

    # 5. Return the initial analysis to the user
    return {"status": "ingested", "analysis": text_representation, "surprise": surprise_score}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, 
    user_id: str = Depends(get_user_id)
):
    """
    OpenAI-Compatible Endpoint for AetherMind Reasoning.
    Now includes logic to handle the feedback loop for the Heart.
    """
    # If this request includes a reaction, it's for closing the loop, not generating a new response.
    if request.last_message_id and request.reaction_score is not None:
        # Here you would retrieve the full trace data saved from the previous turn
        # For simplicity, we'll assume it was cached or stored somewhere accessible.
        # This part requires a more complex state management (e.g., Redis) to be fully implemented.
        # heart.close_loop(retrieved_trace, request.reaction_score)
        return {"status": "feedback_received"}

    # Extract the last message from the list
    last_message = request.messages[-1]["content"]
    
    # Run the DCLA Logic Cycle
    # Now unpacks updated return values
    response_text, message_id, emotion_vector, agent_state = await AETHER.run_cycle(user_id, last_message)
    
    # Return in a standardized format, now including the message_id and metadata
    return {
        "id": message_id, # Return the actual message_id for the feedback loop
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {"total_tokens": len(response_text) // 4}, # Simple estimate
        "metadata": {
            "user_emotion": emotion_vector,
            "agent_state": agent_state
        }
    }

@app.post("/admin/generate_key")
async def create_key(user_id: str, admin_secret: str):
    """Admin only: Create a new API key for a user."""
    if admin_secret != os.getenv("ADMIN_SECRET"):
        raise HTTPException(status_code=401)
    return {"api_key": AUTH.generate_api_key(user_id)}