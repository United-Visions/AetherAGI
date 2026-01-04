#!/usr/bin/env python3
"""
SDK API Endpoint Testing Script
Tests all /v1/* endpoints with live API key
"""

import os
import sys
import json
from aethermind import AetherMindClient

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_success(message):
    print(f"{GREEN}✓ {message}{RESET}")

def print_error(message):
    print(f"{RED}✗ {message}{RESET}")

def print_info(message):
    print(f"{BLUE}ℹ {message}{RESET}")

def print_warning(message):
    print(f"{YELLOW}⚠ {message}{RESET}")

def test_sdk_endpoints():
    """Test all SDK endpoints with live API"""
    
    # Get API key from environment
    api_key = os.getenv("AETHERMIND_API_KEY")
    if not api_key:
        print_error("AETHERMIND_API_KEY not found in environment")
        print_info("Set it with: export AETHERMIND_API_KEY=am_live_...")
        return False
    
    print_info(f"Testing with API key: {api_key[:20]}...")
    
    # Initialize client
    try:
        client = AetherMindClient(api_key=api_key)
        print_success("Client initialized successfully")
    except Exception as e:
        print_error(f"Failed to initialize client: {e}")
        return False
    
    # Test 1: SDK Chat Endpoint
    print("\n" + "="*60)
    print_info("TEST 1: /v1/chat endpoint")
    print("="*60)
    try:
        response = client.chat(
            message="What is the meaning of life?",
            namespace="universal"
        )
        print_success("Chat endpoint working!")
        print(f"  Answer: {response['answer'][:100]}...")
        print(f"  Confidence: {response.get('confidence', 'N/A')}")
        print(f"  Tokens: {response.get('tokens_used', 'N/A')}")
    except Exception as e:
        print_error(f"Chat endpoint failed: {e}")
    
    # Test 2: Memory Search Endpoint
    print("\n" + "="*60)
    print_info("TEST 2: /v1/memory/search endpoint")
    print("="*60)
    try:
        results = client.search_memory(
            query="life philosophy wisdom",
            namespace="universal",
            top_k=5
        )
        print_success("Memory search endpoint working!")
        print(f"  Found {len(results)} results")
        if results:
            print(f"  Top result: {results[0].get('text', '')[:80]}...")
            print(f"  Score: {results[0].get('score', 'N/A')}")
    except Exception as e:
        print_error(f"Memory search endpoint failed: {e}")
    
    # Test 3: ToolForge Endpoint
    print("\n" + "="*60)
    print_info("TEST 3: /v1/tools/create endpoint")
    print("="*60)
    try:
        tool_result = client.create_tool(
            name="weather_checker",
            description="Check current weather for a city",
            code="""
def check_weather(city: str):
    # Mock implementation
    return f"Weather in {city}: Sunny, 72°F"
            """,
            parameters={
                "city": {"type": "string", "description": "City name"}
            }
        )
        print_success("ToolForge endpoint working!")
        print(f"  Tool ID: {tool_result.get('tool_id', 'N/A')}")
        print(f"  Status: {tool_result.get('status', 'N/A')}")
    except Exception as e:
        print_error(f"ToolForge endpoint failed: {e}")
    
    # Test 4: Usage Endpoint
    print("\n" + "="*60)
    print_info("TEST 4: /v1/usage endpoint")
    print("="*60)
    try:
        usage = client.get_usage()
        print_success("Usage endpoint working!")
        print(f"  Requests remaining: {usage.get('requests_remaining', 'N/A')}")
        print(f"  Plan: {usage.get('plan', 'N/A')}")
        print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
        print(f"  Reset at: {usage.get('reset_at', 'N/A')}")
    except Exception as e:
        print_error(f"Usage endpoint failed: {e}")
    
    # Test 5: Namespaces Endpoint
    print("\n" + "="*60)
    print_info("TEST 5: /v1/namespaces endpoint")
    print("="*60)
    try:
        namespaces = client.list_namespaces()
        print_success("Namespaces endpoint working!")
        print(f"  Available namespaces: {', '.join(namespaces)}")
    except Exception as e:
        print_error(f"Namespaces endpoint failed: {e}")
    
    # Test 6: Knowledge Cartridge Endpoint
    print("\n" + "="*60)
    print_info("TEST 6: /v1/knowledge/cartridge endpoint")
    print("="*60)
    try:
        cartridge_result = client.create_knowledge_cartridge(
            name="test_cartridge",
            namespace="universal",
            documents=[
                "AetherMind is a digital organism designed for continuous learning.",
                "The architecture separates reasoning from knowledge storage.",
                "Phase 1 focuses on linguistic genesis and text-based interaction."
            ],
            metadata={"source": "test", "version": "1.0"}
        )
        print_success("Knowledge cartridge endpoint working!")
        print(f"  Cartridge ID: {cartridge_result.get('cartridge_id', 'N/A')}")
        print(f"  Status: {cartridge_result.get('status', 'N/A')}")
        print(f"  Documents processed: {cartridge_result.get('documents_processed', 'N/A')}")
    except Exception as e:
        print_error(f"Knowledge cartridge endpoint failed: {e}")
    
    print("\n" + "="*60)
    print_success("All SDK endpoint tests completed!")
    print("="*60)
    return True


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║         AetherMind SDK Endpoint Testing Suite         ║
    ║                  Live API Tests                        ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    success = test_sdk_endpoints()
    sys.exit(0 if success else 1)
