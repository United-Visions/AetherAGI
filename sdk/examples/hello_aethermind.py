#!/usr/bin/env python3
"""
Hello AetherMind! 🚀
Simple example demonstrating AetherMind AGI integration
"""

import os
from aethermind import AetherMindClient

def main():
    print("=" * 60)
    print("🧠 Hello AetherMind - Python SDK Demo")
    print("=" * 60)
    print()
    
    # Get API key from environment or use demo
    api_key = os.getenv("AETHERMIND_API_KEY", "am_live_demo_key")
    
    if api_key == "am_live_demo_key":
        print("⚠️  Using demo API key. Set AETHERMIND_API_KEY environment variable")
        print("   to use your own key from https://aethermind.ai")
        print()
    
    # Initialize client
    print("📡 Connecting to AetherMind AGI...")
    try:
        client = AetherMindClient(api_key=api_key)
        print("✅ Connected!\n")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Example 1: Basic chat
    print("=" * 60)
    print("Example 1: Basic Chat")
    print("=" * 60)
    
    question = "What is the difference between AI and AGI?"
    print(f"💬 Question: {question}")
    print()
    
    try:
        response = client.chat(question)
        print(f"🤖 AetherMind: {response['answer']}")
        
        if response.get('confidence'):
            print(f"📊 Confidence: {response['confidence']:.0%}")
        
        if response.get('reasoning_steps'):
            print(f"\n🧠 Reasoning Process:")
            for i, step in enumerate(response['reasoning_steps'], 1):
                print(f"   {i}. {step}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Example 2: Domain specialist (Legal)
    print("=" * 60)
    print("Example 2: Legal Domain Specialist")
    print("=" * 60)
    
    legal_question = "What are the key elements of a valid contract?"
    print(f"⚖️  Legal Question: {legal_question}")
    print()
    
    try:
        response = client.chat(legal_question, namespace="legal")
        print(f"🤖 AetherMind (Legal): {response['answer'][:300]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Example 3: Memory search
    print("=" * 60)
    print("Example 3: Infinite Memory Search")
    print("=" * 60)
    
    search_query = "AGI capabilities"
    print(f"🔍 Searching memory for: {search_query}")
    print()
    
    try:
        memories = client.search_memory(
            query=search_query,
            top_k=3,
            include_episodic=True
        )
        
        if memories:
            print(f"📚 Found {len(memories)} relevant memories:")
            for i, memory in enumerate(memories, 1):
                print(f"\n   {i}. {memory['text'][:200]}...")
                print(f"      Score: {memory['score']:.2f} | Time: {memory['timestamp']}")
        else:
            print("   No memories found (try chatting first!)")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Example 4: Usage statistics
    print("=" * 60)
    print("Example 4: Usage & Rate Limits")
    print("=" * 60)
    
    try:
        usage = client.get_usage()
        print(f"📊 API Usage Statistics:")
        print(f"   Plan: {usage.get('plan', 'N/A')}")
        print(f"   Requests Remaining: {usage.get('requests_remaining', 'N/A')}")
        print(f"   Total Tokens Used: {usage.get('total_tokens', 'N/A')}")
        print(f"   Reset At: {usage.get('reset_at', 'N/A')}")
    except Exception as e:
        print(f"⚠️  Usage stats unavailable: {e}")
    
    print()
    print("=" * 60)
    print("✨ Demo Complete!")
    print("=" * 60)
    print()
    print("📖 Learn more:")
    print("   - Documentation: https://aethermind.ai/documentation")
    print("   - GitHub: https://github.com/United-Visions/AetherAGI")
    print("   - Discord: https://discord.gg/aethermind")
    print()
    print("🚀 Ready to build with real AGI? Get your API key at:")
    print("   https://aethermind.ai")
    print()

if __name__ == "__main__":
    main()
