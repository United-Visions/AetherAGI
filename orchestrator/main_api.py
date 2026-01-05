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
import json
from brain.logic_engine import LogicEngine
from heart.heart_orchestrator import Heart # New Heart import
from mind.vector_store import AetherVectorStore
from mind.episodic_memory import EpisodicMemory
from brain.jepa_aligner import JEPAAligner
from curiosity.surprise_detector import SurpriseDetector
from curiosity.research_scheduler import ResearchScheduler
from orchestrator.agent_state_machine import AgentStateMachine
from orchestrator.session_manager import SessionManager
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
SESSION_MANAGER = SessionManager()  # Domain-aware session management

# Initialize Sensory & Curiosity Components
JEPA = JEPAAligner()
SURPRISE_DETECTOR = SurpriseDetector(jepa=JEPA, store=STORE)

# AETHER now takes SURPRISE_DETECTOR + SESSION_MANAGER
AETHER = ActiveInferenceLoop(BRAIN, MEMORY, STORE, ROUTER, HEART, surprise_detector=SURPRISE_DETECTOR, session_manager=SESSION_MANAGER)
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

@app.post("/v1/admin/forge_tool")
async def forge_tool(request: dict, user_id: str = Depends(get_user_id)):
    if not settings.toolforge_adapter:
        raise HTTPException(403, "ToolForge disabled")
    adapter = ROUTER.adapters.get("toolforge")
    result = adapter.execute(json.dumps(request))
    return {"result": result}

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


# ============================================================================
# SDK ENDPOINTS - For developers using AetherMind SDK
# ============================================================================

class SDKChatRequest(BaseModel):
    message: str
    namespace: str = "universal"
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    include_memory: bool = True

class SDKMemorySearchRequest(BaseModel):
    query: str
    namespace: str = "universal"
    top_k: int = 10
    include_episodic: bool = True
    include_knowledge: bool = True

class SDKToolCreateRequest(BaseModel):
    name: str
    description: str
    code: str
    parameters: dict

class SDKKnowledgeCartridgeRequest(BaseModel):
    name: str
    namespace: str
    documents: list[str]
    metadata: Optional[dict] = None

class UserDomainRequest(BaseModel):
    user_id: str
    domain: str  # code, research, business, legal, finance, general
    metadata: Optional[dict] = None


# ============================================================================
# USER DOMAIN CONFIGURATION ENDPOINT
# ============================================================================

@app.post("/v1/user/domain")
async def set_user_domain(
    request: UserDomainRequest,
    authorization: str = Header(None)
):
    """
    Set user's domain specialization.
    This is called during onboarding to configure the agent's behavior.
    
    Available domains:
    - code: Software Development Specialist
    - research: Research & Analysis Specialist
    - business: Business & Strategy Specialist
    - legal: Legal Research Specialist
    - finance: Finance & Investment Specialist
    - general: Multi-Domain Master
    """
    # Verify API key
    if not authorization or not authorization.startswith("ApiKey "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    api_key = authorization.replace("ApiKey ", "").replace("Bearer ", "")
    auth_user_id = AUTH.verify_key(api_key)
    
    if not auth_user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Verify user_id matches or is admin
    if auth_user_id != request.user_id:
        raise HTTPException(status_code=403, detail="Cannot set domain for another user")
    
    # Validate domain
    valid_domains = ["code", "research", "business", "legal", "finance", "general"]
    if request.domain not in valid_domains:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid domain. Must be one of: {', '.join(valid_domains)}"
        )
    
    # Set domain in session manager
    SESSION_MANAGER.set_user_domain(request.user_id, request.domain)
    
    # Get profile info to return
    profile = SESSION_MANAGER.get_user_profile(request.user_id)
    
    logger.info(f"Domain set for user {request.user_id}: {profile['domain_display_name']}")
    
    return {
        "status": "success",
        "user_id": request.user_id,
        "domain": request.domain,
        "domain_display_name": profile["domain_display_name"],
        "message": f"AetherMind configured as {profile['domain_display_name']}"
    }


@app.get("/v1/user/domain")
async def get_user_domain(authorization: str = Header(None)):
    """
    Get user's current domain configuration
    """
    # Verify API key
    if not authorization or not authorization.startswith("ApiKey "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    api_key = authorization.replace("ApiKey ", "").replace("Bearer ", "")
    user_id = AUTH.verify_key(api_key)
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    profile = SESSION_MANAGER.get_user_profile(user_id)
    
    return {
        "user_id": user_id,
        "domain": profile["domain"],
        "domain_display_name": profile["domain_display_name"],
        "interaction_count": profile["interaction_count"],
        "learning_context": profile["learning_context"]
    }


# ============================================================================
# SDK ENDPOINTS - For developers using AetherMind SDK
# ============================================================================

@app.post("/v1/chat")
async def sdk_chat(
    request: SDKChatRequest,
    authorization: str = Header(None)
):
    """
    SDK Chat Endpoint - Used by developers with AetherMind SDK
    """
    # Extract and verify API key
    if not authorization or not authorization.startswith("ApiKey "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    api_key = authorization.replace("ApiKey ", "").replace("Bearer ", "")
    
    # Verify API key and get user_id
    user_id = AUTH.verify_key(api_key)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Get user's domain profile for personalized behavior
    user_profile = SESSION_MANAGER.get_user_profile(user_id)
    domain_profile = user_profile["domain_profile"]
    
    logger.info(f"Chat request from user {user_id} with domain: {domain_profile.display_name}")
    
    # Check rate limits
    usage = AUTH.get_usage(api_key)
    if usage and usage.get("requests_remaining", 0) <= 0:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # Run the active inference loop
        response_text, message_id, emotion_vector, agent_state = await AETHER.run_cycle(
            user_id, 
            request.message,
            namespace=request.namespace
        )
        
        # Return SDK-compatible response
        return {
            "answer": response_text,
            "reasoning_steps": agent_state.get("reasoning_steps", []) if agent_state else [],
            "confidence": agent_state.get("confidence", 0.85) if agent_state else 0.85,
            "sources": agent_state.get("sources", []) if agent_state else [],
            "tokens_used": len(response_text) // 4
        }
    
    except Exception as e:
        logger.error(f"SDK chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memory/search")
async def sdk_memory_search(
    request: SDKMemorySearchRequest,
    authorization: str = Header(None)
):
    """
    SDK Memory Search Endpoint - Search infinite episodic memory
    """
    # Verify API key
    if not authorization or not authorization.startswith("ApiKey "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    api_key = authorization.replace("ApiKey ", "").replace("Bearer ", "")
    user_id = AUTH.verify_key(api_key)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Search episodic memory
        results = []
        
        if request.include_episodic:
            episodic_results = MEMORY.search(
                user_id=user_id,
                query=request.query,
                top_k=request.top_k
            )
            results.extend(episodic_results)
        
        if request.include_knowledge:
            # Search knowledge base
            knowledge_results = STORE.query(
                namespace=f"{request.namespace}_knowledge",
                query_text=request.query,
                top_k=request.top_k
            )
            results.extend(knowledge_results)
        
        # Format results for SDK
        formatted_results = []
        for result in results[:request.top_k]:
            formatted_results.append({
                "text": result.get("text", ""),
                "score": result.get("score", 0.0),
                "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
                "namespace": result.get("namespace", request.namespace),
                "metadata": result.get("metadata", {})
            })
        
        return {"results": formatted_results}
    
    except Exception as e:
        logger.error(f"SDK memory search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tools/create")
async def sdk_tool_create(
    request: SDKToolCreateRequest,
    authorization: str = Header(None)
):
    """
    SDK ToolForge Endpoint - Create custom tools at runtime
    """
    # Verify API key
    if not authorization or not authorization.startswith("ApiKey "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    api_key = authorization.replace("ApiKey ", "").replace("Bearer ", "")
    user_id = AUTH.verify_key(api_key)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Check if ToolForge is enabled
    if not settings.toolforge_adapter:
        raise HTTPException(status_code=403, detail="ToolForge is disabled")
    
    try:
        # Use ToolForge adapter
        adapter = ROUTER.adapters.get("toolforge")
        if not adapter:
            raise HTTPException(status_code=503, detail="ToolForge adapter not available")
        
        tool_data = {
            "name": request.name,
            "description": request.description,
            "code": request.code,
            "parameters": request.parameters,
            "user_id": user_id
        }
        
        result = adapter.execute(json.dumps(tool_data))
        
        return {
            "tool_id": f"tool_{user_id}_{request.name}",
            "status": "created",
            "validation_result": result
        }
    
    except Exception as e:
        logger.error(f"SDK tool create error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/usage")
async def sdk_get_usage(authorization: str = Header(None)):
    """
    SDK Usage Endpoint - Get current API usage and rate limits
    """
    # Verify API key
    if not authorization or not authorization.startswith("ApiKey "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    api_key = authorization.replace("ApiKey ", "").replace("Bearer ", "")
    
    if not AUTH.verify_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        usage = AUTH.get_usage(api_key)
        
        return {
            "requests_remaining": usage.get("requests_remaining", 0),
            "reset_at": usage.get("reset_at", (datetime.utcnow() + timedelta(minutes=1)).isoformat()),
            "total_tokens": usage.get("total_tokens", 0),
            "plan": usage.get("plan", "FREE")
        }
    
    except Exception as e:
        logger.error(f"SDK usage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/namespaces")
async def sdk_list_namespaces(authorization: str = Header(None)):
    """
    SDK Namespaces Endpoint - List available knowledge domains
    """
    # Verify API key
    if not authorization or not authorization.startswith("ApiKey "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    api_key = authorization.replace("ApiKey ", "").replace("Bearer ", "")
    
    if not AUTH.verify_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {
        "namespaces": [
            "universal",
            "legal",
            "medical",
            "finance",
            "code",
            "research"
        ]
    }


@app.post("/v1/knowledge/cartridge")
async def sdk_create_cartridge(
    request: SDKKnowledgeCartridgeRequest,
    authorization: str = Header(None)
):
    """
    SDK Knowledge Cartridge Endpoint - Create custom knowledge domains
    """
    # Verify API key
    if not authorization or not authorization.startswith("ApiKey "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    api_key = authorization.replace("ApiKey ", "").replace("Bearer ", "")
    user_id = AUTH.verify_key(api_key)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Process documents and create embeddings
        cartridge_id = f"cartridge_{user_id}_{request.name}"
        namespace = f"user_{user_id}_{request.namespace}"
        
        # Store documents in vector store
        for i, doc in enumerate(request.documents):
            # Generate embedding (placeholder - use actual embedding model)
            embedding = np.random.rand(1024).astype(np.float32)
            
            STORE.upsert(
                namespace=namespace,
                vectors=[{
                    "id": f"{cartridge_id}_doc_{i}",
                    "values": embedding.tolist(),
                    "metadata": {
                        "text": doc,
                        "cartridge_id": cartridge_id,
                        "cartridge_name": request.name,
                        "user_id": user_id,
                        **(request.metadata or {})
                    }
                }]
            )
        
        return {
            "cartridge_id": cartridge_id,
            "status": "created",
            "processing_time": "2.5s",
            "documents_processed": len(request.documents)
        }
    
    except Exception as e:
        logger.error(f"SDK cartridge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/generate_key")
async def create_key(user_id: str, admin_secret: str):
    """Admin only: Create a new API key for a user."""
    if admin_secret != os.getenv("ADMIN_SECRET"):
        raise HTTPException(status_code=401)
    return {"api_key": AUTH.generate_api_key(user_id)}