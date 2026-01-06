"""
Aether client for benchmark mode.

This client calls Aether with a special system prompt that:
1. Disables all action tags (<aether-*>, <think>, etc.)
2. Forces clean, direct answers
3. Uses the correct answer format for the benchmark
"""
import os
import re
import time
import asyncio
import logging
from typing import Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# LiteLLM looks for GEMINI_API_KEY, but .env has GOOGLE_API_KEY
# Set the expected key if we have the Google key
if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# HuggingFace datasets looks for HF_TOKEN, but .env has HUGGINGFACE_API_KEY
if os.getenv("HUGGINGFACE_API_KEY") and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY")

from typing import Optional

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# System prompt that forces benchmark mode
BENCHMARK_MODE_SYSTEM_PROMPT = """You are being evaluated on a benchmark test. 

CRITICAL RULES:
1. Output ONLY your final answer(s) - no explanations, no reasoning, no tags
2. Do NOT use any XML tags like <think>, <aether-write>, <aether-sandbox>, etc.
3. Do NOT explain your work - just give the answer
4. Do NOT say "I think" or "The answer is" - just output the answer itself
5. If multiple questions are provided, provide the answers in a numbered list corresponding to the question numbers.

ANSWER FORMAT: {answer_format_instructions}
"""


ANSWER_FORMAT_INSTRUCTIONS = {
    "number": "Output only the numerical answer. For batched questions, provide one number per line. Example:\n1. 42\n2. 100",
    "letter": "Output only the letter (A, B, C, D, etc.). For batched questions, provide one letter per line. Example:\n1. B\n2. C",
    "code": "Output only the Python code inside ```python``` blocks. For batched questions, separate them clearly.",
    "text": "Output only the answer text, no explanations.",
    "json": "Output only valid JSON.",
}

GOOD_EXAMPLES = {
    "number": "42\n-15.5\n1000000",
    "letter": "A\nB\nC",
    "code": "```python\ndef solve(x):\n    return x * 2\n```",
    "text": "Paris\nAlbert Einstein\n1969",
    "json": '{"function": "search", "args": {"query": "test"}}',
}

BAD_EXAMPLES = {
    "number": "<think>Let me calculate...</think> The answer is 42.\nI believe it's 42 because...\n#### 42 (after working through the problem)",
    "letter": "I think the answer is B because...\n<aether-research>Looking up...</aether-research> B\nThe correct choice is B.",
    "code": "Here's my solution:\n<aether-write path='sol.py'>code</aether-write>\nThe code above solves it.",
    "text": "Let me think... <think>reasoning</think> The answer is Paris.\nBased on my knowledge, I believe it's Paris because...",
    "json": 'Here is the JSON: {"function": "search"}\n<think>I formatted it as JSON</think>',
}


class AetherBenchmarkClient:
    """
    Client for calling Aether in benchmark mode.
    
    Can use either:
    1. Direct LiteLLM calls (faster, uses same model as Aether)
    2. HTTP calls to Aether API (uses full Aether pipeline)
    """
    
    def __init__(
        self,
        mode: str = "litellm",  # "litellm" or "api"
        api_base: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        model: str = "gemini/gemini-2.5-pro",
        fallback_models: list = None,
    ):
        self.mode = mode
        self.api_base = api_base
        self.api_key = api_key or os.getenv("AETHERMIND_API_KEY")
        self.model = model
        # Updated to Google's latest 2.5 suite
        self.fallback_models = fallback_models or ["gemini/gemini-2.5-flash", "gemini/gemini-1.5-pro"]
        
        if mode == "litellm" and not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for litellm mode. Install with: pip install litellm")
        if mode == "api" and not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for api mode. Install with: pip install httpx")
    
    def _build_system_prompt(self, answer_format: str) -> str:
        """Build the benchmark mode system prompt for a specific answer format."""
        return BENCHMARK_MODE_SYSTEM_PROMPT.format(
            answer_format_instructions=ANSWER_FORMAT_INSTRUCTIONS.get(answer_format, "Output only your answer."),
            good_examples=GOOD_EXAMPLES.get(answer_format, "Just the answer"),
            bad_examples=BAD_EXAMPLES.get(answer_format, "Explanations or tags before/after the answer"),
        )
    
    def _strip_all_tags(self, response: str) -> str:
        """Strip any remaining tags from response (safety net)."""
        if response is None:
            return ""
        
        # Ensure it's a string
        response = str(response)
        
        # Remove aether-* tags
        response = re.sub(r'<aether-[^>]*>.*?</aether-[^>]*>', '', response, flags=re.DOTALL)
        response = re.sub(r'<aether-[^>]*/>', '', response)
        
        # Remove think tags
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove any XML-like tags
        response = re.sub(r'</?[a-zA-Z][a-zA-Z0-9_-]*[^>]*>', '', response)
        
        # Clean up whitespace
        response = re.sub(r'\n{3,}', '\n\n', response)
        return response.strip()
    
    async def ask(
        self,
        question: str,
        answer_format: str = "text",
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Ask Aether a benchmark question.
        
        Args:
            question: The question to ask
            answer_format: Expected format ("number", "letter", "code", "text", "json")
            timeout: Request timeout in seconds
            
        Returns:
            Dict with keys: response, latency_ms, tokens_used, raw_response
        """
        start_time = time.time()
        system_prompt = self._build_system_prompt(answer_format)
        
        if self.mode == "litellm":
            result = await self._ask_litellm(question, system_prompt, timeout)
        else:
            result = await self._ask_api(question, system_prompt, answer_format, timeout)
        
        # Strip any tags that leaked through
        result["response"] = self._strip_all_tags(result["raw_response"])
        result["latency_ms"] = (time.time() - start_time) * 1000
        
        return result
    
    async def _ask_litellm(
        self,
        question: str,
        system_prompt: str,
        timeout: float,
    ) -> Dict[str, Any]:
        """Call LLM directly via LiteLLM."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        
        try:
            # Set a high retry count specifically for the 429 rate limits on Gemini Flash Exp
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                timeout=timeout,
                fallbacks=self.fallback_models,
                num_retries=3,
                retry_strategy="exponential_backoff_retry"
            )
            
            return {
                "raw_response": response.choices[0].message.content or "ERROR: Empty response from model",
                "tokens_used": response.usage.total_tokens if response.usage else 0,
            }
        except Exception as e:
            return {
                "raw_response": f"ERROR: {str(e)}",
                "tokens_used": 0,
            }
    
    async def _ask_api(
        self,
        question: str,
        system_prompt: str,
        answer_format: str,
        timeout: float,
    ) -> Dict[str, Any]:
        """Call Aether via HTTP API."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    f"{self.api_base}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                        "metadata": {
                            "benchmark_mode": True,
                            "answer_format": answer_format,
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                return {
                    "raw_response": data["choices"][0]["message"]["content"] or "ERROR: Empty response from API",
                    "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                }
            except Exception as e:
                return {
                    "raw_response": f"ERROR: {str(e)}",
                    "tokens_used": 0,
                }
    
    def ask_sync(
        self,
        question: str,
        answer_format: str = "text",
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for ask()."""
        return asyncio.run(self.ask(question, answer_format, timeout))


class PlainLLMClient:
    """
    Plain LLM client for generating new benchmark questions.
    
    This does NOT use benchmark mode - it allows the LLM to be creative
    when generating new problems.
    """
    
    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash-exp",
        fallback_models: list = None,
    ):
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required. Install with: pip install litellm")
        
        self.model = model
        self.fallback_models = fallback_models or ["gemini/gemini-1.5-pro-002", "gemini/gemini-1.5-flash-002"]
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        """Generate text with the LLM."""
        logger.info(f"🤖 PlainLLMClient: Generating with model={self.model}")
        logger.debug(f"   Prompt length: {len(prompt)} chars, temp={temperature}, max_tokens={max_tokens}")
        
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                fallbacks=self.fallback_models,
            )
            content = response.choices[0].message.content
            
            if not content:
                error_msg = "Empty response from LLM (content was None or empty)"
                logger.error(f"❌ {error_msg}")
                print(f"❌ LLM ERROR: {error_msg}")
                return f"ERROR: {error_msg}"
            
            logger.info(f"✅ LLM response received: {len(content)} chars")
            logger.debug(f"   Response preview: {content[:200]}...")
            return content
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ LLM generation failed: {error_msg}")
            print(f"❌ LLM ERROR: {error_msg}")
            
            # Log more details for common errors
            if "api_key" in error_msg.lower():
                print("   💡 Check your GEMINI_API_KEY or GOOGLE_API_KEY in .env")
            elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                print("   💡 Rate limit hit - try again later or use a different model")
            elif "timeout" in error_msg.lower():
                print("   💡 Request timed out - the model may be overloaded")
            
            return f"ERROR: {error_msg}"
    
    def generate_sync(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Synchronous wrapper."""
        return asyncio.run(self.generate(prompt, temperature, max_tokens))
