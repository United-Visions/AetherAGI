"""
Path: orchestrator/main_api.py
Role: Production API Gateway. Handles authentication and routing.
"""

import os
from fastapi import FastAPI, Header, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

from .active_inference import ActiveInferenceLoop
from .auth_manager import AuthManager
from .router import Router
from brain.logic_engine import LogicEngine
from mind.vector_store import AetherVectorStore
from mind.episodic_memory import EpisodicMemory

load_dotenv()

app = FastAPI(title="AetherMind AGI API", version="1.0.0")

# --- CORS Middleware ---
# This allows your frontend (running on a different domain) to make API calls to this backend.
origins = [
    "https://aethermind-frontend.onrender.com",
    "http://localhost:5000", # For local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Setup Security
API_KEY_NAME = "X-Aether-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Initialize Core Components
STORE = AetherVectorStore(api_key=os.getenv("PINECONE_API_KEY"))
MEMORY = EpisodicMemory(STORE)
BRAIN = LogicEngine(runpod_key=os.getenv("RUNPOD_API_KEY"), endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"), pinecone_key=os.getenv("PINECONE_API_KEY"))
ROUTER = Router()
AETHER = ActiveInferenceLoop(BRAIN, MEMORY, STORE, ROUTER)
AUTH = AuthManager()

# --- Schemas ---
class ChatCompletionRequest(BaseModel):
    model: str = "aethermind-v1"
    messages: list
    user: str # The user_id for episodic memory tracking

# --- Middleware-like Auth ---
async def get_user_id(api_key: str = Security(api_key_header)):
    user_id = AUTH.verify_key(api_key)
    if not user_id:
        raise HTTPException(status_code=403, detail="Invalid or missing Aether-Secret-Key")
    return user_id

# --- Endpoints ---

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, 
    user_id: str = Depends(get_user_id)
):
    """
    OpenAI-Compatible Endpoint for AetherMind Reasoning.
    """
    # Extract the last message from the list (standard OpenAI format)
    last_message = request.messages[-1]["content"]
    
    # Run the DCLA Logic Cycle
    response_text = await AETHER.run_cycle(user_id, last_message)
    
    # Return in a standardized format
    return {
        "id": "aether-gen-123",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {"total_tokens": len(response_text) // 4} # Simple estimate
    }

@app.post("/admin/generate_key")
async def create_key(user_id: str, admin_secret: str):
    """Admin only: Create a new API key for a user."""
    if admin_secret != os.getenv("ADMIN_SECRET"):
        raise HTTPException(status_code=401)
    return {"api_key": AUTH.generate_api_key(user_id)}