"""
AetherMind SDK Data Models
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """User subscription tiers"""
    FREE = "FREE"
    PRO = "PRO"
    ENTERPRISE = "ENTERPRISE"
    ADMIN = "ADMIN"


class ChatMessage(BaseModel):
    """Chat message structure"""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class MemoryQuery(BaseModel):
    """Memory search query"""
    query: str = Field(..., description="Search query text")
    namespace: str = Field(default="universal", description="Target namespace")
    top_k: int = Field(default=10, description="Number of results")
    include_episodic: bool = Field(default=True, description="Include conversation history")
    include_knowledge: bool = Field(default=True, description="Include learned knowledge")


class ToolCall(BaseModel):
    """Tool invocation structure"""
    name: str = Field(..., description="Tool name")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")
    description: Optional[str] = Field(None, description="What the tool does")


class MemoryResult(BaseModel):
    """Memory search result"""
    text: str = Field(..., description="Retrieved text")
    score: float = Field(..., description="Relevance score 0.0-1.0")
    timestamp: str = Field(..., description="When this was stored")
    namespace: str = Field(..., description="Source namespace")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional data")


class ChatResponse(BaseModel):
    """Chat API response"""
    answer: str = Field(..., description="AetherMind's response")
    reasoning_steps: Optional[List[str]] = Field(None, description="Thought process")
    confidence: Optional[float] = Field(None, description="Confidence score 0.0-1.0")
    sources: Optional[List[str]] = Field(None, description="Knowledge sources used")
    tokens_used: Optional[int] = Field(None, description="Total tokens consumed")
