# AetherMind SDKs

Official software development kits for integrating AetherMind AGI into your applications.

## 🚀 Quick Start

Choose your platform:

### Python
```bash
pip install aethermind
```
```python
from aethermind import AetherMindClient

client = AetherMindClient(api_key="am_live_your_key")
response = client.chat("What is AGI?")
print(response["answer"])
```

### JavaScript/TypeScript
```bash
npm install @aethermind/sdk
```
```typescript
import { AetherMindClient } from '@aethermind/sdk';

const client = new AetherMindClient({ apiKey: 'am_live_your_key' });
const response = await client.chat({ message: 'What is AGI?' });
console.log(response.answer);
```

### React/Next.js
```bash
npm install @aethermind/sdk
```
```tsx
import { AetherMindClient } from '@aethermind/sdk';

const client = new AetherMindClient({
  apiKey: process.env.NEXT_PUBLIC_AETHERMIND_API_KEY
});

export default function Chat() {
  const [response, setResponse] = useState('');
  
  const handleSend = async (message: string) => {
    const res = await client.chat({ message });
    setResponse(res.answer);
  };
  
  return <ChatInterface onSend={handleSend} response={response} />;
}
```

## 📦 Available SDKs

| Platform | Package | Status | Docs |
|----------|---------|--------|------|
| Python | `aethermind` | ✅ Ready | [README](./python/README.md) |
| JavaScript/TypeScript | `@aethermind/sdk` | ✅ Ready | [README](./javascript/README.md) |
| React Native | `@aethermind/sdk` | ✅ Ready | [Docs](../docs/integration/API_AUTHENTICATION.md) |
| Flutter/Dart | `aethermind` | 🚧 Coming Soon | - |
| Go | - | 📅 Planned | - |
| Ruby | - | 📅 Planned | - |

## 🛠️ SDK Features

All SDKs provide:

- ✅ **Chat API** - Send messages, get responses with reasoning
- ✅ **Memory Search** - Query infinite episodic memory
- ✅ **ToolForge** - Create custom tools at runtime
- ✅ **Namespaces** - Domain-specific knowledge (legal, medical, finance, etc.)
- ✅ **Knowledge Cartridges** - Load custom knowledge bases
- ✅ **Usage Tracking** - Monitor API usage and rate limits
- ✅ **Type Safety** - Full TypeScript/Python type hints
- ✅ **Error Handling** - Comprehensive exception classes
- ✅ **Streaming** - Real-time response streaming
- ✅ **Authentication** - API keys, JWT tokens, sessions

## 📚 Documentation

- [Complete API Documentation](../frontend_flask/templates/documentation.html)
- [Integration Examples](../docs/integration/)
- [SDK Distribution Guide](../SDK_DISTRIBUTION_GUIDE.md)
- [Python SDK Publishing](./python/PUBLISHING.md)

## 🔧 Building SDKs Locally

### Automated Setup

Run the setup script to build and test all SDKs:

```bash
./setup_sdk.sh
```

### Manual Build

#### Python
```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install build
python -m build
pip install dist/aethermind-*.whl
```

#### JavaScript
```bash
cd javascript
npm install
npm run build
npm link
```

## 🚢 Publishing SDKs

### Python → PyPI
```bash
cd python
pip install twine
twine upload --repository testpypi dist/*  # Test first
twine upload dist/*                        # Production
```

### JavaScript → npm
```bash
cd javascript
npm login
npm publish --access public
```

## 🧪 Testing

### Python
```bash
cd python
pip install -e ".[dev]"
pytest
```

### JavaScript
```bash
cd javascript
npm test
```

## 📖 Usage Examples

### Basic Chat
```python
# Python
from aethermind import AetherMindClient

client = AetherMindClient(api_key="am_live_your_key")
response = client.chat("Explain quantum computing")
print(response["answer"])
print(f"Confidence: {response['confidence']}")
```

```typescript
// TypeScript
import { AetherMindClient } from '@aethermind/sdk';

const client = new AetherMindClient({ apiKey: 'am_live_your_key' });
const response = await client.chat({ message: 'Explain quantum computing' });
console.log(response.answer);
console.log(`Confidence: ${response.confidence}`);
```

### Memory Search
```python
# Python
memories = client.search_memory(
    query="quantum physics discussions",
    top_k=5,
    include_episodic=True
)

for memory in memories:
    print(f"{memory['text']} (score: {memory['score']})")
```

```typescript
// TypeScript
const memories = await client.searchMemory({
  query: 'quantum physics discussions',
  topK: 5,
  includeEpisodic: true
});

memories.forEach(m => {
  console.log(`${m.text} (score: ${m.score})`);
});
```

### Domain Specialists
```python
# Python - Legal specialist
legal_response = client.chat(
    "Draft a non-disclosure agreement",
    namespace="legal"
)

# Medical specialist
medical_response = client.chat(
    "Explain Type 2 diabetes symptoms",
    namespace="medical"
)
```

```typescript
// TypeScript - Finance specialist
const financeResponse = await client.chat({
  message: 'Calculate compound interest formula',
  namespace: 'finance'
});

// Code specialist
const codeResponse = await client.chat({
  message: 'Write a binary search in Python',
  namespace: 'code'
});
```

### Custom Tools (ToolForge)
```python
# Python
weather_tool = client.create_tool(
    name="get_weather",
    description="Fetch current weather for a city",
    code="""
def get_weather(city: str) -> dict:
    import requests
    response = requests.get(f"https://api.weather.com/{city}")
    return response.json()
""",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        }
    }
)

# Now AetherMind can use this tool
response = client.chat("What's the weather in Tokyo?")
```

### Framework Integrations

#### Flask
```python
from flask import Flask, request, jsonify
from aethermind import AetherMindClient

app = Flask(__name__)
client = AetherMindClient(api_key="am_live_your_key")

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json["message"]
    response = client.chat(message)
    return jsonify(response)
```

#### Express.js
```typescript
import express from 'express';
import { AetherMindClient } from '@aethermind/sdk';

const app = express();
const client = new AetherMindClient({ apiKey: process.env.API_KEY! });

app.post('/chat', async (req, res) => {
  const { message } = req.body;
  const response = await client.chat({ message });
  res.json(response);
});
```

#### Next.js API Route
```typescript
// app/api/chat/route.ts
import { AetherMindClient } from '@aethermind/sdk';
import { NextResponse } from 'next/server';

const client = new AetherMindClient({
  apiKey: process.env.AETHERMIND_API_KEY!
});

export async function POST(request: Request) {
  const { message } = await request.json();
  const response = await client.chat({ message });
  return NextResponse.json(response);
}
```

## 🔐 Authentication

### API Keys (Recommended)
```bash
export AETHERMIND_API_KEY=am_live_your_key_here
```

```python
# Python - auto-loads from environment
client = AetherMindClient()
```

```typescript
// TypeScript - auto-loads from environment
const client = new AetherMindClient({
  apiKey: process.env.AETHERMIND_API_KEY
});
```

### JWT Tokens (Enterprise)
```python
client = AetherMindClient(jwt_token="eyJhbGc...")
```

```typescript
const client = new AetherMindClient({
  jwtToken: 'eyJhbGc...'
});
```

## 💡 Rate Limits

| Plan | Requests/Min | Tokens/Month | Price |
|------|--------------|--------------|-------|
| Free | 100 | 1M | $0 |
| Pro | 1,000 | 50M | $99/mo |
| Enterprise | 10,000 | Unlimited | Custom |

Check usage:
```python
usage = client.get_usage()
print(f"Requests remaining: {usage['requests_remaining']}")
```

## 🐛 Error Handling

```python
from aethermind import AuthenticationError, RateLimitError

try:
    response = client.chat("Hello")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded - upgrade plan")
```

```typescript
import { AuthenticationError, RateLimitError } from '@aethermind/sdk';

try {
  const response = await client.chat({ message: 'Hello' });
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error('Invalid API key');
  } else if (error instanceof RateLimitError) {
    console.error('Rate limit exceeded');
  }
}
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md)

### Development Setup
```bash
# Fork and clone repo
git clone https://github.com/YOUR_USERNAME/AetherAGI.git
cd AetherAGI/sdk

# Setup Python SDK
cd python
pip install -e ".[dev]"
pytest

# Setup JavaScript SDK
cd ../javascript
npm install
npm run build
npm test
```

## 📞 Support

- 📧 Email: [dev@aethermind.ai](mailto:dev@aethermind.ai)
- 💬 Discord: [discord.gg/aethermind](https://discord.gg/aethermind)
- 📖 Docs: [aethermind.ai/documentation](https://aethermind.ai/documentation)
- 🐛 Issues: [GitHub Issues](https://github.com/United-Visions/AetherAGI/issues)

## 📜 License

Apache License 2.0 - See [LICENSE](../LICENSE) file

---

**Built with ❤️ by the AetherMind Team**  
*Real AGI, Not Role-Playing*
