#!/usr/bin/env python3
"""Quick test to verify LLM connectivity for benchmark question generation."""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Check and map keys
google_key = os.getenv('GOOGLE_API_KEY')
gemini_key = os.getenv('GEMINI_API_KEY')

print(f"GOOGLE_API_KEY: {'SET (' + google_key[:15] + '...)' if google_key else 'NOT SET'}")
print(f"GEMINI_API_KEY: {'SET (' + gemini_key[:15] + '...)' if gemini_key else 'NOT SET'}")

if google_key and not gemini_key:
    os.environ['GEMINI_API_KEY'] = google_key
    print("✓ Mapped GOOGLE_API_KEY → GEMINI_API_KEY")

import litellm

async def test_llm():
    print("\nTesting LiteLLM with gemini/gemini-1.5-flash...")
    try:
        resp = await litellm.acompletion(
            model='gemini/gemini-1.5-flash',
            messages=[{'role': 'user', 'content': 'Say hello in exactly 3 words.'}],
            max_tokens=20
        )
        content = resp.choices[0].message.content
        print(f"✓ LLM Response: {content}")
        return True
    except Exception as e:
        print(f"✗ LLM Error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm())
    exit(0 if success else 1)
