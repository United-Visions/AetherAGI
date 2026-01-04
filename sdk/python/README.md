# AetherMind Python SDK

Official Python SDK for **AetherMind AGI** - Real AGI, Not Role-Playing

[![PyPI version](https://badge.fury.io/py/aethermind.svg)](https://badge.fury.io/py/aethermind)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Features

- 🧠 **Real AGI** - Axiom-based learning, not prompt engineering
- 🗄️ **Infinite Memory** - Full episodic recall of all conversations
- 🔧 **ToolForge** - Create custom tools at runtime
- 🎯 **Domain Specialists** - Legal, Medical, Finance, Code, Research
- 🔒 **Enterprise Security** - RBAC, audit logging, encryption
- ⚡ **Fast & Reliable** - Built on RunPod + Pinecone infrastructure

## Installation

```bash
pip install aethermind
```

Or install from source:

```bash
git clone https://github.com/United-Visions/AetherAGI.git
cd AetherAGI/sdk/python
pip install -e .
```

## Quick Start

### 1. Get Your API Key

Sign up at [aethermind.ai](https://aethermind.ai) and create an API key from your dashboard.

### 2. Basic Usage

```python
from aethermind import AetherMindClient

# Initialize client
client = AetherMindClient(api_key="am_live_your_key_here")

# Ask a question
response = client.chat("What is Newton's Second Law?")
print(response["answer"])

# Search memory
memories = client.search_memory("previous discussions about physics")
for memory in memories:
    print(f"{memory['text']} (score: {memory['score']})")
```

### 3. Environment Variables

Create a `.env` file:

```env
AETHERMIND_API_KEY=am_live_your_key_here
AETHERMIND_BASE_URL=https://api.aethermind.ai  # Optional
```

Then use without passing api_key:

```python
from aethermind import AetherMindClient

client = AetherMindClient()  # Reads from AETHERMIND_API_KEY env var
response = client.chat("Hello, AetherMind!")
```

## Core Concepts

### Namespaces (Domain Knowledge)

AetherMind specializes in different domains through **Knowledge Namespaces**:

```python
# Universal knowledge (default)
client.chat("Explain quantum entanglement", namespace="universal")

# Legal specialist
client.chat("Draft a non-disclosure agreement", namespace="legal")

# Medical specialist
client.chat("Explain Type 2 diabetes symptoms", namespace="medical")

# Finance specialist
client.chat("Calculate compound interest formula", namespace="finance")

# Code specialist
client.chat("Write a Python binary search algorithm", namespace="code")

# Research specialist
client.chat("Summarize recent quantum computing papers", namespace="research")
```

### Infinite Episodic Memory

Every conversation is remembered forever:

```python
# AetherMind remembers all past conversations
client.chat("My favorite color is blue")

# Later...
response = client.chat("What's my favorite color?")
# Output: "Your favorite color is blue, based on our conversation."

# Search specific memories
memories = client.search_memory(
    query="color preferences",
    include_episodic=True,
    top_k=5
)
```

### ToolForge (Custom Tools)

Create tools that AetherMind can use:

```python
# Define a custom tool
weather_tool = {
    "name": "get_weather",
    "description": "Fetch current weather for a city",
    "code": """
def get_weather(city: str) -> dict:
    import requests
    response = requests.get(f"https://api.weather.com/v1/{city}")
    return response.json()
""",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
}

# Register the tool
result = client.create_tool(**weather_tool)
print(f"Tool created: {result['tool_id']}")

# Now AetherMind can use it
response = client.chat("What's the weather in San Francisco?")
```

### Knowledge Cartridges

Load custom knowledge domains:

```python
# Create a knowledge cartridge for your company
cartridge = client.create_knowledge_cartridge(
    name="company_policies",
    namespace="custom_hr",
    documents=[
        "https://yourcompany.com/handbook.pdf",
        "Employee handbook text here...",
        "Policy document text..."
    ],
    metadata={"department": "HR", "version": "2.1"}
)

# Query your custom knowledge
response = client.chat(
    "What is our vacation policy?",
    namespace="custom_hr"
)
```

## Advanced Usage

### Streaming Responses

```python
response = client.chat(
    "Explain quantum computing",
    stream=True
)

for chunk in response:
    print(chunk["text"], end="", flush=True)
```

### Context Control

```python
response = client.chat(
    "Continue our discussion about AI safety",
    include_memory=True,  # Include past conversations
    max_tokens=500,
    temperature=0.7  # 0.0 = deterministic, 1.0 = creative
)
```

### Usage Monitoring

```python
usage = client.get_usage()
print(f"Requests remaining: {usage['requests_remaining']}")
print(f"Plan: {usage['plan']}")
print(f"Reset at: {usage['reset_at']}")
```

### Context Manager

```python
with AetherMindClient(api_key="am_live_your_key") as client:
    response = client.chat("Hello!")
    # Connection automatically closed
```

## API Reference

### AetherMindClient

| Method | Description |
|--------|-------------|
| `chat(message, namespace, stream, max_tokens, temperature, include_memory)` | Send a chat message |
| `search_memory(query, namespace, top_k, include_episodic, include_knowledge)` | Search infinite memory |
| `create_tool(name, description, code, parameters)` | Create a custom tool (ToolForge) |
| `create_knowledge_cartridge(name, namespace, documents, metadata)` | Load custom knowledge |
| `get_usage()` | Get current usage stats and rate limits |
| `list_namespaces()` | List available knowledge domains |

### Exceptions

```python
from aethermind import (
    AetherMindError,          # Base exception
    AuthenticationError,      # Invalid API key
    RateLimitError,          # Rate limit exceeded
    ValidationError,         # Invalid request
    NetworkError             # Connection failed
)

try:
    client.chat("Hello")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit hit - upgrade plan or wait")
except AetherMindError as e:
    print(f"Error: {e}")
```

## Examples

### Flask Integration

```python
from flask import Flask, request, jsonify
from aethermind import AetherMindClient

app = Flask(__name__)
client = AetherMindClient(api_key="am_live_your_key")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    response = client.chat(user_message)
    return jsonify(response)

if __name__ == "__main__":
    app.run()
```

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from aethermind import AetherMindClient, AuthenticationError

app = FastAPI()
client = AetherMindClient(api_key="am_live_your_key")

@app.post("/chat")
async def chat(message: str):
    try:
        response = client.chat(message)
        return response
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key")
```

### Django Integration

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from aethermind import AetherMindClient
import json

client = AetherMindClient(api_key="am_live_your_key")

@csrf_exempt
def chat_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        response = client.chat(data["message"])
        return JsonResponse(response)
```

### Async Support

```python
import asyncio
from aethermind import AetherMindClient

async def async_chat():
    client = AetherMindClient(api_key="am_live_your_key")
    
    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        client.chat,
        "What is AGI?"
    )
    return response

# Run
response = asyncio.run(async_chat())
```

## Rate Limits

| Plan | Requests/Min | Tokens/Month | Price |
|------|--------------|--------------|-------|
| Free | 100 | 1M | $0 |
| Pro | 1,000 | 50M | $99/mo |
| Enterprise | 10,000 | Unlimited | Custom |

## Authentication Methods

### 1. API Key (Recommended)

```python
client = AetherMindClient(api_key="am_live_your_key")
```

### 2. Environment Variable

```bash
export AETHERMIND_API_KEY=am_live_your_key
```

```python
client = AetherMindClient()  # Auto-loads from env
```

### 3. JWT Token (Enterprise)

```python
from aethermind import AetherMindClient

client = AetherMindClient(
    jwt_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    base_url="https://api.aethermind.ai"
)
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone repo
git clone https://github.com/United-Visions/AetherAGI.git
cd AetherAGI/sdk/python

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black aethermind/
flake8 aethermind/
```

## Support

- 📧 Email: [dev@aethermind.ai](mailto:dev@aethermind.ai)
- 💬 Discord: [discord.gg/aethermind](https://discord.gg/aethermind)
- 📖 Docs: [aethermind.ai/documentation](https://aethermind.ai/documentation)
- 🐛 Issues: [GitHub Issues](https://github.com/United-Visions/AetherAGI/issues)

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## Roadmap

- ✅ Phase 1: Linguistic Genesis (Chat, Memory, ToolForge)
- 🚧 Phase 2: Sensory Awakening (Vision, JEPA, Multi-modal)
- 📅 Phase 3: Phonetic Articulation (Speech synthesis, voice)
- 📅 Phase 4: Physical Embodiment (Humanoid robotics)

---

**Built with ❤️ by the AetherMind Team**

*Real AGI, Not Role-Playing*
