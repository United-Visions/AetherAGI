"""
AetherMind SDK Client
Main client for interacting with AetherMind AGI API
"""

import httpx
from typing import Optional, Dict, Any, List, Union
import os
from .models import ChatMessage, MemoryQuery, ToolCall, UserRole
from .exceptions import AetherMindError, AuthenticationError, RateLimitError


class AetherMindClient:
    """
    Official Python client for AetherMind AGI
    
    Example:
        >>> from aethermind import AetherMindClient
        >>> client = AetherMindClient(api_key="am_live_your_key")
        >>> response = client.chat("What is Newton's Second Law?")
        >>> print(response["answer"])
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize AetherMind client
        
        Args:
            api_key: Your AetherMind API key (or set AETHERMIND_API_KEY env var)
            base_url: Custom API endpoint (defaults to production)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        self.api_key = api_key or os.getenv("AETHERMIND_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "API key required. Pass api_key parameter or set AETHERMIND_API_KEY environment variable."
            )
        
        self.base_url = base_url or os.getenv("AETHERMIND_BASE_URL", "https://api.aethermind.ai")
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"ApiKey {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"aethermind-python/1.0.0"
            }
        )
    
    def chat(
        self,
        message: str,
        namespace: str = "universal",
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        include_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Send a chat message to AetherMind
        
        Args:
            message: Your message/question
            namespace: Knowledge namespace (universal, legal, medical, finance, code, research)
            stream: Enable streaming responses
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0 = deterministic, 1.0 = creative)
            include_memory: Include episodic memory context
        
        Returns:
            Dict with keys: answer, reasoning_steps, confidence, sources
        
        Raises:
            AuthenticationError: Invalid API key
            RateLimitError: Rate limit exceeded
            AetherMindError: Other API errors
        """
        payload = {
            "message": message,
            "namespace": namespace,
            "stream": stream,
            "include_memory": include_memory
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if temperature:
            payload["temperature"] = temperature
        
        try:
            response = self.client.post("/v1/chat", json=payload)
            self._check_response(response)
            return response.json()
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
    
    def search_memory(
        self,
        query: str,
        namespace: str = "universal",
        top_k: int = 10,
        include_episodic: bool = True,
        include_knowledge: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search AetherMind's infinite memory
        
        Args:
            query: Search query
            namespace: Knowledge namespace to search
            top_k: Number of results to return
            include_episodic: Include conversation history
            include_knowledge: Include learned knowledge
        
        Returns:
            List of memory items with text, score, timestamp, metadata
        """
        payload = {
            "query": query,
            "namespace": namespace,
            "top_k": top_k,
            "include_episodic": include_episodic,
            "include_knowledge": include_knowledge
        }
        
        try:
            response = self.client.post("/v1/memory/search", json=payload)
            self._check_response(response)
            return response.json()["results"]
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
    
    def create_tool(
        self,
        name: str,
        description: str,
        code: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a custom tool for AetherMind (ToolForge)
        
        Args:
            name: Tool name (e.g., "fetch_weather")
            description: What the tool does
            code: Python code for the tool
            parameters: JSON schema for tool parameters
        
        Returns:
            Dict with tool_id, status, validation_result
        """
        payload = {
            "name": name,
            "description": description,
            "code": code,
            "parameters": parameters
        }
        
        try:
            response = self.client.post("/v1/tools/create", json=payload)
            self._check_response(response)
            return response.json()
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
    
    def get_usage(self) -> Dict[str, Any]:
        """
        Get current usage statistics and rate limits
        
        Returns:
            Dict with requests_remaining, reset_at, total_tokens, plan
        """
        try:
            response = self.client.get("/v1/usage")
            self._check_response(response)
            return response.json()
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
    
    def list_namespaces(self) -> List[str]:
        """
        List available knowledge namespaces
        
        Returns:
            List of namespace names
        """
        try:
            response = self.client.get("/v1/namespaces")
            self._check_response(response)
            return response.json()["namespaces"]
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
    
    def create_knowledge_cartridge(
        self,
        name: str,
        namespace: str,
        documents: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a custom knowledge cartridge
        
        Args:
            name: Cartridge name
            namespace: Target namespace
            documents: List of document texts or URLs
            metadata: Additional metadata
        
        Returns:
            Dict with cartridge_id, status, processing_time
        """
        payload = {
            "name": name,
            "namespace": namespace,
            "documents": documents,
            "metadata": metadata or {}
        }
        
        try:
            response = self.client.post("/v1/knowledge/cartridge", json=payload)
            self._check_response(response)
            return response.json()
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
    
    def _check_response(self, response: httpx.Response):
        """Check response for errors"""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded. Upgrade plan or wait for reset.")
            else:
                raise AetherMindError(f"API error: {response.text}")
    
    def _handle_http_error(self, error: httpx.HTTPStatusError):
        """Handle HTTP errors"""
        if error.response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif error.response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        else:
            raise AetherMindError(f"API error: {error.response.text}")
    
    def close(self):
        """Close the HTTP client"""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
