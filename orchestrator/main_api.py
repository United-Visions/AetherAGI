"""
Path: orchestrator/main_api.py
Role: Production API Gateway. Handles authentication and routing.
"""

import os
from dotenv import load_dotenv

# CRITICAL: Load environment variables BEFORE any imports that depend on them
load_dotenv()

from fastapi import FastAPI, Header, HTTPException, Depends, Security, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import numpy as np
import httpx

from loguru import logger

from .active_inference import ActiveInferenceLoop
from .auth_manager_supabase import AuthManagerSupabase
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
from orchestrator.background_worker import BackgroundWorker, set_background_worker
from orchestrator.action_parser import ActionParser
from orchestrator.benchmark_service import BENCHMARK_SERVICE
from perception.eye import Eye
from config.settings import settings
import asyncio
from contextlib import asynccontextmanager

# Track background tasks for cleanup
_background_tasks = []

@asynccontextmanager
async def lifespan(app):
    """Manage application startup and shutdown."""
    logger.info("🚀 AetherMind API starting up...")
    yield
    # Shutdown
    logger.info("🛑 AetherMind API shutting down...")
    # Stop background worker
    if BACKGROUND_WORKER:
        BACKGROUND_WORKER.stop()
    # Cancel any running background tasks
    for task in _background_tasks:
        if not task.done():
            task.cancel()
    logger.info("✅ Shutdown complete")

app = FastAPI(title="AetherMind AGI API", version="1.0.0", lifespan=lifespan)

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

# ============================================================================
# Initialize Core Services (Global Singletons)
# ============================================================================

# Get required API keys from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Authentication Manager
AUTH = AuthManagerSupabase()

# Initialize the perception module (multimodal input)
EYE = Eye()

# Initialize vector store and memory systems
VECTOR_STORE = AetherVectorStore(api_key=PINECONE_API_KEY)
EPISODIC_MEMORY = EpisodicMemory(VECTOR_STORE)

# Initialize brain components
LOGIC_ENGINE = LogicEngine(pinecone_key=PINECONE_API_KEY)
HEART = Heart(pinecone_key=PINECONE_API_KEY)
JEPA_ALIGNER = JEPAAligner()

# Initialize session management
SESSION_MANAGER = SessionManager()
ACTION_PARSER = ActionParser()

# Initialize curiosity and research modules
SURPRISE_DETECTOR = SurpriseDetector(jepa=JEPA_ALIGNER, store=VECTOR_STORE)
RESEARCH_SCHEDULER = ResearchScheduler(redis_url=REDIS_URL)

# Initialize the routing layer with available adapters
ROUTER = Router()

# Initialize the main active inference loop (the "nervous system")
AETHER = ActiveInferenceLoop(
    brain=LOGIC_ENGINE,
    memory=EPISODIC_MEMORY,
    store=VECTOR_STORE,
    router=ROUTER,
    heart=HEART,
    surprise_detector=SURPRISE_DETECTOR,
    session_manager=SESSION_MANAGER
)

# Initialize background worker for autonomous goals
BACKGROUND_WORKER = BackgroundWorker(
    brain=LOGIC_ENGINE,
    heart=HEART,
    store=VECTOR_STORE,
    memory=EPISODIC_MEMORY,
    action_parser=ACTION_PARSER,
    router=ROUTER
)
set_background_worker(BACKGROUND_WORKER)

# Also expose as AUTH_MANAGER for some endpoints that use that name
AUTH_MANAGER = AUTH

logger.info("✅ All core services initialized")

# Setup Security
API_KEY_NAME = "X-Aether-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_optional_api_key(
    x_key: Optional[str] = Header(None, alias="X-Aether-Key"),
    secret_key: Optional[str] = Header(None, alias="Aether-Secret-Key"),
    auth: Optional[str] = Header(None, alias="Authorization")
):
    """Checks multiple possible header locations for the API key."""
    if x_key: return x_key
    if secret_key: return secret_key
    if auth and auth.startswith("Bearer "): return auth.replace("Bearer ", "")
    if auth and auth.startswith("ApiKey "): return auth.replace("ApiKey ", "")
    return None

async def get_user_id(api_key: str = Depends(get_optional_api_key)):
    # Dev mode bypass for local development/benchmarking
    if settings.dev_mode and not api_key:
        # Use first pilot user from settings, or fallback
        pilot_user = settings.pilot_users[0] if settings.pilot_users else "dev_user"
        logger.warning(f"⚠️ DEV MODE: Allowing unauthenticated request as {pilot_user}")
        return pilot_user
    
    if not api_key:
        raise HTTPException(status_code=401, detail="API key missing")
    user_data = AUTH.verify_api_key(api_key)
    if not user_data:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return user_data["user_id"]


async def verify_api_key(api_key: str = Depends(get_optional_api_key)) -> dict:
    """
    Verify API key and return full user data dict.
    Checks multiple header variations (X-Aether-Key, Aether-Secret-Key, Authorization).
    """
    # Dev mode bypass for local development/benchmarking
    if settings.dev_mode and not api_key:
        # Use first pilot user from settings, or fallback
        pilot_user = settings.pilot_users[0] if settings.pilot_users else "dev_user"
        logger.warning(f"⚠️ DEV MODE: Allowing unauthenticated request as {pilot_user}")
        return {"user_id": pilot_user, "role": "admin", "permissions": ["all"]}
    
    if not api_key:
        raise HTTPException(status_code=401, detail="API key missing")
    user_data = AUTH.verify_api_key(api_key)
    if not user_data:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return user_data


# ============================================================================
# Request Models (must be defined before endpoints that use them)
# ============================================================================

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request format."""
    model: str = "aethermind-v1"
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    user: Optional[str] = None
    # AetherMind-specific fields for feedback loop
    last_message_id: Optional[str] = None
    reaction_score: Optional[float] = None
    # Context for special modes (onboarding, etc.)
    context: Optional[dict] = None


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
    Uses the internal Eye module (LiteLLM/Gemini) directly.
    """
    try:
        logger.info(f"Multimodal ingest started: filename={file.filename}, content_type={file.content_type}, user_id={user_id}")
        
        contents = await file.read()
        logger.info(f"File read successfully: size={len(contents)} bytes")
        
        # Use the internal Eye module directly
        text_representation = await EYE.ingest(contents, file.content_type)
        logger.info(f"Eye ingest completed: {text_representation[:100]}...")
        
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
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Multimodal ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, 
    user_id: str = Depends(get_user_id)
):
    """
    OpenAI-Compatible Endpoint for AetherMind Reasoning.
    Now includes logic to handle the feedback loop for the Heart.
    Supports onboarding mode for personalized conversation.
    """
    # If this request includes a reaction, it's for closing the loop, not generating a new response.
    if request.last_message_id and request.reaction_score is not None:
        return {"status": "feedback_received"}

    # Extract the last message from the list
    last_message = request.messages[-1]["content"] if request.messages else ""
    
    # Check for onboarding mode
    context = request.context or {}
    is_onboarding = context.get("isOnboarding", False)
    onboarding_mode = context.get("mode", "")
    current_profile = context.get("currentProfile", {})
    
    # Check for active persona
    active_persona = current_profile.get("activePersona")
    personas = current_profile.get("personas", {})
    
    # Check for system prompt override in the messages list
    override_system = None
    for msg in request.messages:
        if msg.get("role") == "system":
            override_system = msg.get("content")
            break
    
    # If onboarding or welcome_back, inject special system context
    if is_onboarding or onboarding_mode == "welcome_back":
        onboarding_prompt = build_onboarding_prompt(onboarding_mode, current_profile)
        override_system = onboarding_prompt
        logger.info(f"🎭 [CONTEXT] Mode: {onboarding_mode}, Profile facts: {current_profile.get('learnedFacts', {})}")
    
    # If active persona, inject persona prompt
    elif active_persona and active_persona in personas:
        persona_prompt = build_persona_prompt(personas[active_persona], current_profile)
        override_system = persona_prompt
        logger.info(f"🎭 [PERSONA] Active: {active_persona}")

    # Run the DCLA Logic Cycle
    response_text, message_id, emotion_vector, agent_state = await AETHER.run_cycle(
        user_id, last_message, override_prompt=override_system
    )
    
    # Get activity events (tool creation, file changes, code execution)
    activity_events = AETHER.get_activity_events()
    
    # Extract thinking steps for frontend display
    thinking_steps = agent_state.get("thinking_steps", []) if agent_state else []
    
    # Build metadata
    metadata = {
        "user_emotion": emotion_vector,
        "agent_state": agent_state,
        "activity_events": activity_events,
        "reasoning_steps": thinking_steps
    }
    
    # If onboarding, parse response for learned facts and completion signals
    if is_onboarding:
        onboarding_metadata = parse_onboarding_response(response_text, current_profile)
        metadata.update(onboarding_metadata)
    
    # Parse for persona commands in any message
    persona_metadata = parse_persona_commands(last_message, response_text, personas)
    metadata.update(persona_metadata)
    
    # Return in a standardized format
    return {
        "id": message_id,
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {"total_tokens": len(response_text) // 4},
        "metadata": metadata
    }


def build_onboarding_prompt(mode: str, profile: dict) -> str:
    """Build a system prompt for onboarding and welcome conversations."""
    
    learned_facts = profile.get("learnedFacts", {})
    conversation_log = profile.get("conversationLog", [])
    
    # Get user's name if we know it
    name = learned_facts.get("name", "")
    name_context = f"The user's name is {name}. USE THIS NAME - do not make up other names! " if name else ""
    
    # Build context from what we've learned
    facts_context = ""
    if learned_facts:
        facts_list = [f"- {k}: {v}" for k, v in learned_facts.items()]
        facts_context = f"\n\nFACTS I KNOW ABOUT THIS USER (use these!):\n" + "\n".join(facts_list)
    
    # Build conversation history context
    convo_context = ""
    if conversation_log:
        recent = conversation_log[-10:]  # Last 10 messages
        convo_lines = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Me"
            content = msg.get("content", "")[:200]  # Truncate long messages
            convo_lines.append(f"{role}: {content}")
        convo_context = f"\n\nRECENT CONVERSATION (continue from here, don't restart!):\n" + "\n".join(convo_lines)
    
    # Welcome back mode - shorter prompt for returning users
    if mode == "welcome_back":
        return f"""You are AetherMind, a warm and helpful AI assistant.

{name_context}

The user has returned. Give them a brief, warm welcome back and ask what they'd like to work on.
Keep it natural and conversational - one or two sentences max.
{facts_context}

CRITICAL: Use the user's actual name if you know it. Do NOT introduce yourself again or re-ask their name.

Be friendly but not over-the-top. No need for emojis unless it feels natural."""
    
    # Onboarding mode
    base_prompt = f"""You are AetherMind, a warm, curious, and genuinely interested AI assistant.

Your personality:
- Friendly and warm, but not overly enthusiastic or corporate
- Genuinely curious about people - you want to understand who they are
- Patient and adaptive - you adjust to their communication style
- Honest and direct when appropriate
- You speak naturally, like a knowledgeable friend

{name_context}
{facts_context}
{convo_context}

CRITICAL RULES:
1. NEVER re-introduce yourself if you already have in this conversation
2. NEVER ask for the user's name if you already know it
3. Continue the conversation naturally from where it left off
4. If they told you facts, REMEMBER them and use them
5. Don't repeat greetings - if you said hello, move on

ONBOARDING GOALS (only if not already done):
- Learn their name (if not known)
- Understand what they want to do
- Learn how they like to communicate

RESPONSE FORMAT:
- Respond conversationally, be yourself
- When onboarding is complete, end with: [ONBOARDING_COMPLETE]
- If you want to offer photo/video option, end with: [REQUEST_MEDIA]

DO NOT:
- Say "Hello" or "Nice to meet you" if you already did
- Ask their name if you already know it
- Be repetitive or formulaic
- Lose track of what was already discussed

DO NOT:
- Use excessive emojis (one or two is fine)
- Be overly formal or stiff
- Ask multiple questions at once
- Rush through getting to know them"""

    if mode == "onboarding_start":
        base_prompt += "\n\nThis is the FIRST message. Introduce yourself warmly and ask their name."
    elif mode == "onboarding_resume":
        base_prompt += "\n\nThe user has returned mid-conversation. Welcome them back and continue naturally."
    
    return base_prompt


def parse_onboarding_response(response_text: str, profile: dict) -> dict:
    """Parse the agent's response for onboarding signals and learned facts."""
    
    metadata = {}
    
    # Check for completion signal
    if "[ONBOARDING_COMPLETE]" in response_text:
        metadata["onboarding_complete"] = True
        # Clean the marker from response (will be done on frontend too)
    
    # Check for media request
    if "[REQUEST_MEDIA]" in response_text:
        metadata["request_media"] = True
    
    # Try to extract learned facts from the conversation
    # This is a simple heuristic - the agent can also explicitly return facts
    learned_facts = {}
    
    # Look for name patterns in recent user messages
    conversation_log = profile.get("conversationLog", [])
    import re
    
    for msg in conversation_log[-5:]:  # Last 5 messages
        if msg.get("role") == "user":
            content = msg.get("content", "").strip()
            
            # Check if it looks like just a name (all caps or title case, 1-4 words)
            # This catches "DECTRICK ANTONIO MCGEE" or "John Smith"
            words = content.split()
            if 1 <= len(words) <= 4:
                # Check if all words look like name parts (letters only, reasonable length)
                if all(word.isalpha() and 2 <= len(word) <= 15 for word in words):
                    # Looks like a name! Format it nicely
                    learned_facts["name"] = ' '.join(word.capitalize() for word in words)
                    break
            
            # Also try patterns like "I'm X" or "My name is X"
            name_patterns = [
                r"(?:i'm|im|i am|my name is|call me|it's|its|name:?)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
            ]
            for pattern in name_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    name_str = match.group(1).strip()
                    # Format nicely
                    learned_facts["name"] = ' '.join(word.capitalize() for word in name_str.split())
                    break
    
    if learned_facts:
        metadata["learned_facts"] = learned_facts
    
    return metadata


def build_persona_prompt(persona: dict, profile: dict) -> str:
    """Build a system prompt for an active persona."""
    
    name = persona.get("name", "Unknown")
    description = persona.get("description", "")
    traits = persona.get("traits", [])
    speech_style = persona.get("speech_style", "")
    
    # Get user's name
    user_name = profile.get("learnedFacts", {}).get("name", "")
    user_context = f"You're talking to {user_name}. " if user_name else ""
    
    traits_text = "\n".join([f"- {t}" for t in traits]) if traits else ""
    
    return f"""You are now acting as the persona: **{name}**

{description}

{user_context}

Your personality traits:
{traits_text}

Your speech style: {speech_style}

IMPORTANT:
- Stay fully in character as {name}
- Use the speech patterns and mannerisms described
- React to things how {name} would react
- You can still help with tasks, but do it AS this persona
- If the user says "switch to [persona]" or "be yourself" or "back to normal", acknowledge the switch

Remember: You're not just answering questions - you ARE this character right now."""


def parse_persona_commands(user_message: str, response_text: str, existing_personas: dict) -> dict:
    """Parse user messages for persona creation and switching commands."""
    import re
    
    metadata = {}
    user_lower = user_message.lower()
    
    # Check for persona switch commands
    switch_patterns = [
        r"(?:switch to|be|act (?:like|as)|become|change to)\s+['\"]?(\w+(?:\s+\w+)*)['\"]?(?:\s+persona)?",
        r"(?:use|activate)\s+['\"]?(\w+(?:\s+\w+)*)['\"]?\s+(?:persona|mode|personality)",
    ]
    
    for pattern in switch_patterns:
        match = re.search(pattern, user_lower)
        if match:
            persona_name = match.group(1).strip().lower()
            # Check if this persona exists
            for existing_name in existing_personas:
                if existing_name.lower() == persona_name or persona_name in existing_name.lower():
                    metadata["switch_persona"] = existing_name
                    logger.info(f"🎭 [PERSONA] Switching to: {existing_name}")
                    break
    
    # Check for "back to normal" or "be yourself"
    normal_patterns = ["be yourself", "back to normal", "normal mode", "default persona", "no persona"]
    for pattern in normal_patterns:
        if pattern in user_lower:
            metadata["switch_persona"] = None  # Clear active persona
            logger.info("🎭 [PERSONA] Switching back to default")
    
    # Check for persona creation commands
    # Patterns like "save a persona called X" or "create persona: X"
    create_patterns = [
        r"(?:save|create|add|make)\s+(?:a\s+)?persona\s+(?:called|named|:)\s*['\"]?([^'\"]+)['\"]?",
        r"(?:new|add)\s+personality\s*[:\s]+['\"]?([^'\"]+)['\"]?",
    ]
    
    for pattern in create_patterns:
        match = re.search(pattern, user_lower)
        if match:
            persona_name = match.group(1).strip()
            # The response should contain the persona definition
            # Look for traits/description in the conversation
            metadata["creating_persona"] = persona_name
            logger.info(f"🎭 [PERSONA] User wants to create: {persona_name}")
    
    return metadata


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
    user_data = AUTH.verify_api_key(api_key)
    
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Verify user_id matches or is admin
    if user_data["user_id"] != request.user_id:
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
    user_data = AUTH.verify_api_key(api_key)
    
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    user_id = user_data["user_id"]
    profile = SESSION_MANAGER.get_user_profile(user_id)
    
    return {
        "user_id": user_id,
        "domain": profile["domain"],
        "domain_display_name": profile["domain_display_name"],
        "interaction_count": profile["interaction_count"],
        "learning_context": profile["learning_context"]
    }


@app.get("/v1/user/permissions")
async def get_user_permissions(user_id: str = Depends(get_user_id)):
    """
    Get user's role and permissions based on their API key
    Returns: role (free/pro/enterprise/admin) and list of permissions
    """
    # Get API key from header to check role
    from fastapi import Request
    request = Request(scope={"type": "http"})
    api_key = request.headers.get("x-aether-key")
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    user_data = AUTH.verify_api_key(api_key)
    
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {
        "user_id": user_data["user_id"],
        "role": user_data["role"],
        "permissions": user_data["permissions"],
        "rate_limit": AUTH.RATE_LIMITS.get(user_data["role"], 100)
    }


# ============================================================================
# USER PROFILE ENDPOINTS - Shell UI Onboarding & Preferences
# ============================================================================

class UserProfileRequest(BaseModel):
    name: Optional[str] = None
    preferences: Optional[dict] = {}
    domain: Optional[str] = None
    conversationStyle: Optional[str] = "friendly"
    goals: Optional[list] = []
    timezone: Optional[str] = None
    mediaOptIn: Optional[bool] = False
    # Persona vector - saved personalities
    personas: Optional[dict] = {}
    activePersona: Optional[str] = None
    # Learned facts from conversation
    learnedFacts: Optional[dict] = {}
    onboarded: Optional[bool] = False
    conversationLog: Optional[list] = []
    askLater: Optional[list] = []

# In-memory profile storage (would be Supabase in production)
USER_PROFILES = {}

@app.get("/v1/user/profile")
async def get_user_profile(user_id: str = Depends(get_user_id)):
    """
    Get user's full profile including onboarding status and personas
    """
    profile = USER_PROFILES.get(user_id, {})
    
    # Check if pilot user
    is_pilot = user_id in settings.pilot_users
    
    # Merge with session data
    session_profile = SESSION_MANAGER.get_user_profile(user_id)
    
    return {
        "user_id": user_id,
        "onboarded": profile.get("onboarded", False),
        "name": profile.get("name") or profile.get("learnedFacts", {}).get("name"),
        "preferences": profile.get("preferences", {}),
        "domain": profile.get("domain", "general"),
        "conversationStyle": profile.get("conversationStyle", "friendly"),
        "goals": profile.get("goals", []),
        "timezone": profile.get("timezone"),
        "mediaOptIn": profile.get("mediaOptIn", False),
        "personas": profile.get("personas", {}),
        "activePersona": profile.get("activePersona"),
        "learnedFacts": profile.get("learnedFacts", {}),
        "conversationLog": profile.get("conversationLog", []),
        "askLater": profile.get("askLater", []),
        "is_pilot": is_pilot,
        "interaction_count": session_profile.get("interaction_count", 0)
    }

@app.post("/v1/user/profile")
async def save_user_profile(
    profile_data: UserProfileRequest,
    user_id: str = Depends(get_user_id)
):
    """
    Save user profile from onboarding or settings update.
    Supports incremental saves - merges with existing data.
    """
    # Get existing profile
    existing = USER_PROFILES.get(user_id, {})
    
    # Merge personas - don't overwrite, add to
    existing_personas = existing.get("personas", {})
    new_personas = profile_data.personas or {}
    merged_personas = {**existing_personas, **new_personas}
    
    # Merge learned facts
    existing_facts = existing.get("learnedFacts", {})
    new_facts = profile_data.learnedFacts or {}
    merged_facts = {**existing_facts, **new_facts}
    
    # Update with new data
    updated = {
        **existing,
        "name": profile_data.name or merged_facts.get("name") or existing.get("name"),
        "preferences": {**existing.get("preferences", {}), **(profile_data.preferences or {})},
        "domain": profile_data.domain or existing.get("domain", "general"),
        "conversationStyle": profile_data.conversationStyle or existing.get("conversationStyle", "friendly"),
        "goals": profile_data.goals or existing.get("goals", []),
        "timezone": profile_data.timezone or existing.get("timezone"),
        "mediaOptIn": profile_data.mediaOptIn if profile_data.mediaOptIn is not None else existing.get("mediaOptIn", False),
        "personas": merged_personas,
        "activePersona": profile_data.activePersona if profile_data.activePersona is not None else existing.get("activePersona"),
        "learnedFacts": merged_facts,
        "onboarded": profile_data.onboarded if profile_data.onboarded is not None else existing.get("onboarded", False),
        "conversationLog": profile_data.conversationLog or existing.get("conversationLog", []),
        "askLater": profile_data.askLater or existing.get("askLater", []),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    USER_PROFILES[user_id] = updated
    
    # Also update session manager
    if profile_data.domain:
        SESSION_MANAGER.set_user_domain(user_id, profile_data.domain)
    
    logger.info(f"Profile saved for user {user_id}: {updated.get('name', 'Unknown')}, personas: {list(merged_personas.keys())}")
    
    return {
        "status": "success",
        "user_id": user_id,
        "profile": updated
    }


# ============================================================================
# PERSONA MANAGEMENT ENDPOINTS
# ============================================================================

class PersonaRequest(BaseModel):
    name: str
    description: str
    traits: list[str] = []
    speech_style: str = "Natural and conversational"

@app.post("/v1/personas/create")
async def create_persona(
    persona: PersonaRequest,
    user_id: str = Depends(get_user_id)
):
    """
    Create a new persona for the user.
    Personas are saved to the user's profile and can be switched between.
    """
    # Get existing profile
    profile = USER_PROFILES.get(user_id, {})
    personas = profile.get("personas", {})
    
    # Create the persona
    new_persona = {
        "name": persona.name,
        "description": persona.description,
        "traits": persona.traits,
        "speech_style": persona.speech_style,
        "created_at": datetime.utcnow().isoformat()
    }
    
    personas[persona.name] = new_persona
    profile["personas"] = personas
    USER_PROFILES[user_id] = profile
    
    logger.info(f"🎭 Persona created: {persona.name} for user {user_id}")
    
    return {
        "status": "success",
        "persona": new_persona,
        "all_personas": list(personas.keys())
    }

@app.get("/v1/personas")
async def list_personas(user_id: str = Depends(get_user_id)):
    """Get all personas for the current user"""
    profile = USER_PROFILES.get(user_id, {})
    personas = profile.get("personas", {})
    active = profile.get("activePersona")
    
    return {
        "personas": personas,
        "active_persona": active,
        "count": len(personas)
    }

@app.post("/v1/personas/switch/{persona_name}")
async def switch_persona(
    persona_name: str,
    user_id: str = Depends(get_user_id)
):
    """Switch to a different persona or back to default"""
    profile = USER_PROFILES.get(user_id, {})
    personas = profile.get("personas", {})
    
    if persona_name.lower() in ["none", "default", "normal"]:
        profile["activePersona"] = None
        USER_PROFILES[user_id] = profile
        return {"status": "success", "active_persona": None, "message": "Switched to default personality"}
    
    if persona_name not in personas:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_name}' not found")
    
    profile["activePersona"] = persona_name
    USER_PROFILES[user_id] = profile
    
    logger.info(f"🎭 Switched to persona: {persona_name} for user {user_id}")
    
    return {
        "status": "success",
        "active_persona": persona_name,
        "persona": personas[persona_name]
    }

@app.delete("/v1/personas/{persona_name}")
async def delete_persona(
    persona_name: str,
    user_id: str = Depends(get_user_id)
):
    """Delete a persona"""
    profile = USER_PROFILES.get(user_id, {})
    personas = profile.get("personas", {})
    
    if persona_name not in personas:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_name}' not found")
    
    del personas[persona_name]
    
    # Clear active if it was the deleted one
    if profile.get("activePersona") == persona_name:
        profile["activePersona"] = None
    
    profile["personas"] = personas
    USER_PROFILES[user_id] = profile
    
    return {
        "status": "success",
        "message": f"Persona '{persona_name}' deleted",
        "remaining_personas": list(personas.keys())
    }


# ============================================================================
# BACKGROUND TASKS ENDPOINTS
# ============================================================================

class TaskStatusRequest(BaseModel):
    task_ids: list[str]

# In-memory task storage
BACKGROUND_TASKS = {}

@app.post("/v1/tasks/status")
async def get_tasks_status(
    request: TaskStatusRequest,
    user_id: str = Depends(get_user_id)
):
    """
    Get status of background tasks
    """
    tasks = []
    for task_id in request.task_ids:
        if task_id in BACKGROUND_TASKS:
            task = BACKGROUND_TASKS[task_id]
            if task.get("user_id") == user_id:
                tasks.append(task)
    
    return {"tasks": tasks}

@app.post("/v1/tasks/create")
async def create_background_task(
    task_type: str,
    task_name: str,
    task_data: dict,
    user_id: str = Depends(get_user_id)
):
    """
    Create a new background task
    """
    task_id = f"task_{datetime.utcnow().timestamp()}_{user_id[:8]}"
    
    task = {
        "id": task_id,
        "type": task_type,
        "name": task_name,
        "status": "pending",
        "progress": 0,
        "user_id": user_id,
        "data": task_data,
        "created_at": datetime.utcnow().isoformat()
    }
    
    BACKGROUND_TASKS[task_id] = task
    
    # If it's a goal, add to background worker
    if task_type == "goal":
        await BACKGROUND_WORKER.goal_tracker.create_goal(
            user_id=user_id,
            description=task_name,
            goal_type=task_data.get("goal_type", "general")
        )
    
    return task


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


# ==========================================
# AUTONOMOUS GOAL MANAGEMENT ENDPOINTS
# ==========================================

class CreateGoalRequest(BaseModel):
    """Request model for creating autonomous goals."""
    description: str
    priority: int = 5  # 1-10
    metadata: Optional[dict] = None


class GoalStatusResponse(BaseModel):
    """Response model for goal status."""
    goal_id: str
    status: str
    description: str
    progress: dict
    subtasks: list
    created_at: str
    updated_at: str


@app.post("/v1/goals/create")
async def create_autonomous_goal(
    request: CreateGoalRequest,
    user_data: dict = Depends(verify_api_key)
):
    """
    Create a new autonomous goal that will be completed in the background.
    
    The goal will:
    - Be automatically decomposed into subtasks
    - Execute independently without user interaction
    - Continue working even if browser is closed
    - Resume after server restarts
    - Self-heal from errors and retry failed steps
    
    Example:
        POST /v1/goals/create
        {
            "description": "Create a Flask todo app with SQLite database",
            "priority": 8,
            "metadata": {"domain": "code", "complexity": "medium"}
        }
    """
    try:
        user_id = user_data["user_id"]
        
        # Submit goal to background worker
        goal_id = await BACKGROUND_WORKER.submit_goal(
            user_id=user_id,
            description=request.description,
            priority=request.priority,
            metadata=request.metadata or {}
        )
        
        logger.info(f"Goal created: {goal_id} for user {user_id}")
        
        return {
            "goal_id": goal_id,
            "status": "pending",
            "message": "Goal submitted. It will be processed autonomously in the background.",
            "check_status_url": f"/v1/goals/{goal_id}/status"
        }
    
    except Exception as e:
        logger.error(f"Failed to create goal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/goals/{goal_id}/status")
async def get_goal_status(
    goal_id: str,
    user_data: dict = Depends(verify_api_key)
):
    """
    Get the current status of an autonomous goal.
    
    Returns:
    - Current status (pending, in_progress, completed, failed)
    - Progress percentage
    - List of subtasks with their statuses
    - Any error messages
    """
    try:
        user_id = user_data["user_id"]
        
        # Get goal from tracker
        goal = await BACKGROUND_WORKER.goal_tracker.get_goal(goal_id)
        
        # Verify user owns this goal
        if goal.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to view this goal")
        
        progress = goal.get_progress()
        
        return {
            "goal_id": goal.goal_id,
            "status": goal.status.value,
            "description": goal.description,
            "progress": progress,
            "subtasks": [
                {
                    "subtask_id": st.subtask_id,
                    "description": st.description,
                    "status": st.status.value,
                    "attempt_count": st.attempt_count,
                    "error_message": st.error_message,
                    "execution_result": st.execution_result
                }
                for st in goal.subtasks
            ],
            "created_at": goal.created_at,
            "updated_at": goal.updated_at
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get goal status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/goals/list")
async def list_user_goals(
    user_data: dict = Depends(verify_api_key),
    status_filter: Optional[str] = None
):
    """
    List all goals for the authenticated user.
    
    Query params:
    - status_filter: Optional filter by status (pending, in_progress, completed, failed)
    """
    try:
        user_id = user_data["user_id"]
        
        # Get all user goals
        all_goals = await BACKGROUND_WORKER.goal_tracker.get_pending_goals(user_id)
        
        # Filter by status if requested
        if status_filter:
            all_goals = [g for g in all_goals if g.status.value == status_filter]
        
        return {
            "goals": [
                {
                    "goal_id": goal.goal_id,
                    "description": goal.description,
                    "status": goal.status.value,
                    "priority": goal.priority,
                    "progress": goal.get_progress(),
                    "created_at": goal.created_at,
                    "updated_at": goal.updated_at
                }
                for goal in all_goals
            ],
            "total": len(all_goals)
        }
    
    except Exception as e:
        logger.error(f"Failed to list goals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# BENCHMARK MANAGEMENT ENDPOINTS
# ==========================================

@app.get("/v1/benchmarks")
async def list_benchmarks(user_data: dict = Depends(verify_api_key)):
    """List all available benchmarks and their last results."""
    return await BENCHMARK_SERVICE.get_available_benchmarks()

@app.post("/v1/benchmarks/run")
async def run_benchmark(
    family: str, 
    model: Optional[str] = None,
    user_data: dict = Depends(verify_api_key)
):
    """Start a benchmark run."""
    benchmark_id = await BENCHMARK_SERVICE.start_benchmark(
        family=family,
        user_id=user_data["user_id"],
        mode="api",
        model=model
    )
    return {"benchmark_id": benchmark_id, "status": "started"}

@app.get("/v1/benchmarks/{benchmark_id}/status")
async def get_benchmark_status(
    benchmark_id: str,
    user_data: dict = Depends(verify_api_key)
):
    """Get status of a running benchmark."""
    return await BENCHMARK_SERVICE.get_status(benchmark_id)

# ==========================================
# PROACTIVE INTERACTION ENDPOINT
# ==========================================

@app.get("/v1/proactive/check")
async def check_proactive_interaction(
    user_data: dict = Depends(verify_api_key)
):
    """
    Check if AetherMind has anything proactive to say to the user.
    This is called when the UI initializes or periodically.
    """
    user_id = user_data["user_id"]
    
    # Run the proactive check through the inference loop
    need_to_speak, message, priority, metadata = await AETHER.check_proactive_need(user_id)
    
    return {
        "should_interact": need_to_speak,
        "message": message,
        "priority": priority,
        "metadata": metadata
    }

# ==========================================
# GitHub Repos API Endpoints
# ==========================================

@app.get("/api/user/repos")
async def get_user_repos(user_data: dict = Depends(verify_api_key)):
    """
    Get list of GitHub repositories the user has connected to AetherMind.
    Returns repos from user metadata in Supabase.
    """
    try:
        user_id = user_data["user_id"]
        
        # Try to get user profile from Supabase
        try:
            if hasattr(AUTH_MANAGER, 'supabase') and AUTH_MANAGER.supabase:
                response = AUTH_MANAGER.supabase.table("users").select("metadata").eq("user_id", user_id).execute()
                
                if response.data:
                    user_metadata = response.data[0].get("metadata", {})
                    connected_repos = user_metadata.get("connected_repos", [])
                    return {"repos": connected_repos}
        except Exception as db_error:
            logger.warning(f"Could not fetch repos from database: {db_error}")
        
        # Return empty list if no repos found or database unavailable
        return {"repos": []}
    
    except Exception as e:
        logger.error(f"Failed to fetch user repos: {e}", exc_info=True)
        # Return empty list instead of error to prevent frontend crash
        return {"repos": []}


# ==========================================
# Apps Management Endpoints
# ==========================================

@app.get("/v1/apps/list")
async def list_user_apps(user_data: dict = Depends(verify_api_key)):
    """
    List all apps created by the authenticated user.
    Returns apps from user metadata or sandbox sessions.
    """
    try:
        user_id = user_data["user_id"]
        
        # Try to get apps from Supabase
        try:
            if hasattr(AUTH_MANAGER, 'supabase') and AUTH_MANAGER.supabase:
                response = AUTH_MANAGER.supabase.table("user_apps").select("*").eq("user_id", user_id).execute()
                
                if response.data:
                    return {
                        "apps": [
                            {
                                "id": app.get("id"),
                                "name": app.get("name", "Unnamed App"),
                                "description": app.get("description", ""),
                                "status": app.get("status", "stopped"),
                                "template": app.get("template", "blank"),
                                "created_at": app.get("created_at"),
                                "updated_at": app.get("updated_at")
                            }
                            for app in response.data
                        ],
                        "total": len(response.data)
                    }
        except Exception as db_error:
            logger.warning(f"Could not fetch apps from database: {db_error}")
        
        # Return empty list if no apps found or database unavailable
        return {"apps": [], "total": 0}
    
    except Exception as e:
        logger.error(f"Failed to list apps: {e}", exc_info=True)
        return {"apps": [], "total": 0}


# =============================================
# UNIFIED PROJECT CREATION API
# Supports: apps, tools, mcp servers, APIs
# =============================================

@app.post("/v1/projects/create")
async def create_project(
    project_data: dict,
    user_data: dict = Depends(verify_api_key)
):
    """
    Create a new project (app, tool, mcp server, or API).
    
    Body:
    {
        "name": "my-project",
        "project_type": "app|tool|mcp|api",
        "template": "blank|react|cli|stdio|fastapi|...",
        "description": "What the project does",
        "preview_mode": "iframe|terminal|logs|api-tester"
    }
    """
    try:
        user_id = user_data["user_id"]
        project_type = project_data.get("project_type", "app")
        
        # Validate project type
        valid_types = ["app", "tool", "mcp", "api"]
        if project_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid project_type. Must be one of: {valid_types}")
        
        project_record = {
            "id": f"{project_type}_{user_id}_{int(datetime.now().timestamp())}",
            "user_id": user_id,
            "name": project_data.get("name", f"New {project_type.upper()}"),
            "project_type": project_type,
            "template": project_data.get("template", "blank"),
            "description": project_data.get("description", ""),
            "preview_mode": project_data.get("preview_mode", "iframe"),
            "status": "created",
            "sandbox_id": None,
            "files": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Try to save to Supabase
        try:
            if hasattr(AUTH_MANAGER, 'supabase') and AUTH_MANAGER.supabase:
                AUTH_MANAGER.supabase.table("user_projects").insert(project_record).execute()
        except Exception as db_error:
            logger.warning(f"Could not save project to database: {db_error}")
        
        logger.info(f"Created {project_type} project '{project_record['name']}' for user {user_id}")
        return {"project": project_record, "message": f"{project_type.upper()} project created successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create project: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/projects/list")
async def list_projects(
    project_type: Optional[str] = None,
    user_data: dict = Depends(verify_api_key)
):
    """
    List all projects for the authenticated user.
    Optionally filter by project_type (app, tool, mcp, api).
    """
    try:
        user_id = user_data["user_id"]
        
        # Try to fetch from Supabase
        try:
            if hasattr(AUTH_MANAGER, 'supabase') and AUTH_MANAGER.supabase:
                query = AUTH_MANAGER.supabase.table("user_projects").select("*").eq("user_id", user_id)
                if project_type:
                    query = query.eq("project_type", project_type)
                response = query.order("created_at", desc=True).execute()
                return {"projects": response.data, "total": len(response.data)}
        except Exception as db_error:
            logger.warning(f"Could not fetch projects from database: {db_error}")
        
        return {"projects": [], "total": 0}
    
    except Exception as e:
        logger.error(f"Failed to list projects: {e}", exc_info=True)
        return {"projects": [], "total": 0}


@app.post("/v1/apps/create")
async def create_app(
    app_data: dict,
    user_data: dict = Depends(verify_api_key)
):
    """
    Create a new app for the authenticated user.
    (Legacy endpoint - redirects to unified /v1/projects/create)
    """
    try:
        user_id = user_data["user_id"]
        
        app_record = {
            "id": f"app_{user_id}_{int(datetime.now().timestamp())}",
            "user_id": user_id,
            "name": app_data.get("name", "New App"),
            "description": app_data.get("description", ""),
            "template": app_data.get("template", "blank"),
            "project_type": "app",
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Try to save to Supabase
        try:
            if hasattr(AUTH_MANAGER, 'supabase') and AUTH_MANAGER.supabase:
                AUTH_MANAGER.supabase.table("user_projects").insert(app_record).execute()
        except Exception as db_error:
            logger.warning(f"Could not save app to database: {db_error}")
        
        return {"app": app_record, "message": "App created successfully"}
    
    except Exception as e:
        logger.error(f"Failed to create app: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/user/repos/connect")
async def connect_repo(
    repo_data: dict,
    user_data: dict = Depends(verify_api_key)
):
    """
    Connect a GitHub repository to user's account.
    
    Body:
    {
        "name": "repo-name",
        "full_name": "owner/repo-name",
        "description": "Repo description",
        "private": false,
        "html_url": "https://github.com/owner/repo"
    }
    """
    try:
        user_id = user_data["user_id"]
        
        # Get current repos
        response = AUTH_MANAGER.supabase.table("users").select("metadata").eq("user_id", user_id).execute()
        
        user_metadata = response.data[0].get("metadata", {}) if response.data else {}
        connected_repos = user_metadata.get("connected_repos", [])
        
        # Check if already connected
        if any(r["full_name"] == repo_data["full_name"] for r in connected_repos):
            return {"message": "Repository already connected", "repos": connected_repos}
        
        # Add new repo
        new_repo = {
            "name": repo_data["name"],
            "full_name": repo_data["full_name"],
            "description": repo_data.get("description", ""),
            "private": repo_data.get("private", False),
            "html_url": repo_data.get("html_url", f"https://github.com/{repo_data['full_name']}"),
            "connected_at": datetime.utcnow().isoformat()
        }
        connected_repos.append(new_repo)
        
        # Update user metadata
        user_metadata["connected_repos"] = connected_repos
        AUTH_MANAGER.supabase.table("users").update({"metadata": user_metadata}).eq("user_id", user_id).execute()
        
        logger.info(f"Connected repo {repo_data['full_name']} for user {user_id}")
        
        return {
            "message": "Repository connected successfully",
            "repo": new_repo,
            "repos": connected_repos
        }
    
    except Exception as e:
        logger.error(f"Failed to connect repo: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/user/repos/{repo_full_name}")
async def disconnect_repo(
    repo_full_name: str,
    user_data: dict = Depends(verify_api_key)
):
    """
    Disconnect a GitHub repository from user's account.
    repo_full_name should be URL-encoded (e.g., owner%2Frepo)
    """
    try:
        user_id = user_data["user_id"]
        
        # Decode the repo full name
        from urllib.parse import unquote
        repo_full_name = unquote(repo_full_name)
        
        # Get current repos
        response = AUTH_MANAGER.supabase.table("users").select("metadata").eq("user_id", user_id).execute()
        
        user_metadata = response.data[0].get("metadata", {}) if response.data else {}
        connected_repos = user_metadata.get("connected_repos", [])
        
        # Remove repo
        connected_repos = [r for r in connected_repos if r["full_name"] != repo_full_name]
        user_metadata["connected_repos"] = connected_repos
        
        # Update user metadata
        AUTH_MANAGER.supabase.table("users").update({"metadata": user_metadata}).eq("user_id", user_id).execute()
        
        logger.info(f"Disconnected repo {repo_full_name} for user {user_id}")
        
        return {
            "message": "Repository disconnected successfully",
            "repos": connected_repos
        }
    
    except Exception as e:
        logger.error(f"Failed to disconnect repo: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# GitHub OAuth Endpoints (Stub Implementation)
# ============================================================================

@app.get("/auth/github")
async def github_oauth_redirect():
    """
    Initiate GitHub OAuth flow.
    TODO: Implement full OAuth flow with GitHub App credentials.
    """
    # For now, return a message that OAuth needs to be configured
    # In production, this would redirect to:
    # https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&scope=repo,user
    
    return {
        "message": "GitHub OAuth not yet configured",
        "instructions": "To enable GitHub integration, create a GitHub OAuth App and add GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET to your .env file",
        "redirect_url": "https://github.com/settings/developers"
    }


@app.get("/auth/github/callback")
async def github_oauth_callback(code: str):
    """
    Handle GitHub OAuth callback.
    TODO: Exchange code for access token and fetch user repos.
    """
    return {
        "message": "GitHub OAuth callback - not yet implemented",
        "code": code
    }


# ============================================================================
# BODY ADAPTER SWITCHING DEMO
# Proves the same Brain can operate through different Bodies
# ============================================================================

@app.get("/v1/body/list")
async def list_body_adapters():
    """
    List all available Body adapters.
    This demonstrates the modular nature of AetherMind's architecture.
    """
    return {
        "adapters": AETHER.router.get_available_adapters(),
        "description": "Same Brain, different Bodies - switch adapters to control different domains"
    }


@app.post("/v1/body/switch")
async def body_switch_demo(
    body_request: dict
):
    """
    Demonstrate Body switching - route an intent through different adapters.
    
    Body:
    {
        "adapter": "chat|smart_home|automotive",
        "intent": "JSON command or plain text"
    }
    
    Examples:
    - Chat: {"adapter": "chat", "intent": "Hello world"}
    - Smart Home: {"adapter": "smart_home", "intent": {"action": "query"}}
    - Automotive: {"adapter": "automotive", "intent": {"action": "query"}}
    """
    adapter = body_request.get("adapter", "chat")
    intent = body_request.get("intent", "")
    
    # Convert dict intent to JSON string
    if isinstance(intent, dict):
        intent = json.dumps(intent)
    
    available = AETHER.router.get_available_adapters()
    if adapter not in available:
        raise HTTPException(
            status_code=400, 
            detail=f"Adapter '{adapter}' not available. Options: {available}"
        )
    
    # Route through the specified adapter
    result = AETHER.router.forward_intent(intent, adapter)
    
    return {
        "adapter_used": adapter,
        "result": json.loads(result) if result.startswith("{") else result,
        "proof": f"Same Brain logic, different Body output ({adapter})"
    }


@app.post("/v1/body/demo")
async def full_body_switch_demo():
    """
    Run a complete demo showing the SAME intent processed by DIFFERENT bodies.
    This is the ultimate proof that AetherMind's Brain is body-agnostic.
    """
    test_intent_chat = "What is the status?"
    test_intent_home = json.dumps({"action": "query"})
    test_intent_car = json.dumps({"action": "query"})
    
    results = {}
    
    # Same conceptual request ("tell me status") through 3 different bodies
    results["chat"] = {
        "body": "chat",
        "intent": test_intent_chat,
        "output": AETHER.router.forward_intent(test_intent_chat, "chat")
    }
    
    results["smart_home"] = {
        "body": "smart_home", 
        "intent": test_intent_home,
        "output": json.loads(AETHER.router.forward_intent(test_intent_home, "smart_home"))
    }
    
    results["automotive"] = {
        "body": "automotive",
        "intent": test_intent_car, 
        "output": json.loads(AETHER.router.forward_intent(test_intent_car, "automotive"))
    }
    
    return {
        "demonstration": "Body Adapter Switching",
        "concept": "Same Brain (reasoning) + Different Bodies (output interfaces)",
        "results": results,
        "proof": "Notice how each adapter returns domain-specific responses from the same intent pattern"
    }


# ============================================================================
# Voice Synthesis Endpoints (Edge TTS)
# ============================================================================

from perception.voice_synthesizer import get_voice_synthesizer, VOICE_PROFILES


class VoiceSynthesisRequest(BaseModel):
    """Request for voice synthesis"""
    text: str
    voice_id: Optional[str] = None
    persona: Optional[str] = None
    rate: Optional[str] = "+0%"
    pitch: Optional[str] = "+0Hz"


@app.post("/v1/voice/synthesize")
async def synthesize_voice(
    request: VoiceSynthesisRequest,
    x_api_key: str = Header(None)
):
    """
    Synthesize speech from text using Edge TTS.
    Returns base64-encoded MP3 audio.
    
    - If persona is provided, uses that persona's voice profile
    - If voice_id is provided, overrides persona voice
    - Rate/pitch can fine-tune the voice
    """
    # Verify API key
    user_data = AUTH.verify_api_key(x_api_key)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        synthesizer = get_voice_synthesizer()
        audio_b64 = await synthesizer.synthesize_to_base64(
            text=request.text,
            voice_id=request.voice_id,
            persona=request.persona,
            rate=request.rate or "+0%",
            pitch=request.pitch or "+0Hz"
        )
        
        return {
            "success": True,
            "audio": audio_b64,
            "format": "mp3",
            "voice_used": request.voice_id or (
                VOICE_PROFILES.get(request.persona.lower().replace(" ", "_"), VOICE_PROFILES["default"]).voice_id
                if request.persona else VOICE_PROFILES["default"].voice_id
            )
        }
    except Exception as e:
        logger.error(f"Voice synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice synthesis failed: {str(e)}")


@app.get("/v1/voice/voices")
async def list_voices(
    language: str = "en",
    x_api_key: str = Header(None)
):
    """
    List all available voices, filtered by language.
    
    Language codes: en (English), es (Spanish), fr (French), de (German), etc.
    """
    user_data = AUTH.verify_api_key(x_api_key)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        synthesizer = get_voice_synthesizer()
        voices = await synthesizer.list_voices(language_filter=language)
        
        return {
            "voices": voices,
            "count": len(voices),
            "profiles": {
                name: {
                    "voice_id": config.voice_id,
                    "rate": config.rate,
                    "pitch": config.pitch
                }
                for name, config in VOICE_PROFILES.items()
            }
        }
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))
