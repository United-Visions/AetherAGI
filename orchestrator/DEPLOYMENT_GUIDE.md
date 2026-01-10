# AetherMind FastAPI Backend Deployment Guide

## Overview

The FastAPI backend (`orchestrator/main_api.py`) serves as the API gateway for AetherMind, providing:
- **OpenAI-compatible endpoints** (`/v1/chat/completions`)
- **SDK endpoints** (`/v1/chat`, `/v1/memory/search`, `/v1/tools/create`, etc.)
- **Admin endpoints** (`/v1/admin/*`)
- **Perception service** (`/v1/ingest/multimodal`)

## Architecture

```
Developer's App
      ↓
  SDK Client (Python/JS)
      ↓
  Authorization: ApiKey {key}
      ↓
FastAPI Backend (main_api.py)
      ↓
  ┌─────────────────────────────┐
  │  AETHER (Active Inference)  │
  │  BRAIN (LogicEngine)        │
  │  MEMORY (EpisodicMemory)    │
  │  STORE (VectorStore)        │
  │  HEART (Emotion)            │
  │  ROUTER (Adapters)          │
  └─────────────────────────────┘
```

## Environment Setup

### Required Environment Variables

Create a `.env` file in the root directory:

```bash
# API Keys
PINECONE_API_KEY=your_pinecone_key
RUNPOD_API_KEY=your_runpod_key
ADMIN_SECRET=your_admin_secret

# Optional Services
FIRECRAWL_API_KEY=your_firecrawl_key
PERCEPTION_SERVICE_URL=http://your-perception-service:8000

# Database (if using)
SUPABASE_URL=your_supabase_url
SB_ANON_KEY=your_supabase_key  # renamed from SUPABASE_ANON_KEY for Supabase secrets storage
```

### Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## SDK Endpoints Reference

### 1. POST `/v1/chat`
**Purpose**: Main chat interface for SDK users

**Request**:
```json
{
  "message": "What is quantum mechanics?",
  "namespace": "universal",
  "stream": false,
  "temperature": 0.7,
  "include_memory": true
}
```

**Response**:
```json
{
  "answer": "Quantum mechanics is...",
  "reasoning_steps": ["Step 1...", "Step 2..."],
  "confidence": 0.92,
  "sources": ["source1", "source2"],
  "tokens_used": 150
}
```

**Headers**:
```
Authorization: ApiKey am_live_xxx
```

### 2. POST `/v1/memory/search`
**Purpose**: Search infinite episodic memory

**Request**:
```json
{
  "query": "conversations about physics",
  "namespace": "universal",
  "top_k": 10,
  "include_episodic": true,
  "include_knowledge": true
}
```

**Response**:
```json
{
  "results": [
    {
      "text": "Conversation excerpt...",
      "score": 0.95,
      "timestamp": "2024-01-15T10:30:00Z",
      "namespace": "universal",
      "metadata": {}
    }
  ]
}
```

### 3. POST `/v1/tools/create`
**Purpose**: ToolForge - Create custom tools at runtime

**Request**:
```json
{
  "name": "weather_checker",
  "description": "Check weather for a city",
  "code": "def check_weather(city):\\n    return f'Weather in {city}'",
  "parameters": {
    "city": {"type": "string"}
  }
}
```

**Response**:
```json
{
  "tool_id": "tool_user123_weather_checker",
  "status": "created",
  "validation_result": "success"
}
```

### 4. GET `/v1/usage`
**Purpose**: Get API usage and rate limits

**Response**:
```json
{
  "requests_remaining": 450,
  "reset_at": "2024-01-15T12:00:00Z",
  "total_tokens": 15000,
  "plan": "PRO"
}
```

### 5. GET `/v1/namespaces`
**Purpose**: List available knowledge domains

**Response**:
```json
{
  "namespaces": [
    "universal",
    "legal",
    "medical",
    "finance",
    "code",
    "research"
  ]
}
```

### 6. POST `/v1/knowledge/cartridge`
**Purpose**: Create custom knowledge domains

**Request**:
```json
{
  "name": "company_docs",
  "namespace": "universal",
  "documents": [
    "Document 1 content...",
    "Document 2 content..."
  ],
  "metadata": {
    "source": "internal",
    "version": "1.0"
  }
}
```

**Response**:
```json
{
  "cartridge_id": "cartridge_user123_company_docs",
  "status": "created",
  "processing_time": "2.5s",
  "documents_processed": 2
}
```

## Authentication

All SDK endpoints use the **Authorization header** format:

```
Authorization: ApiKey am_live_xxx
```

Or alternatively:

```
Authorization: Bearer am_live_xxx
```

The backend extracts the key and verifies it using `AuthManager`:

```python
api_key = authorization.replace("ApiKey ", "").replace("Bearer ", "")
user_id = AUTH.verify_key(api_key)
```

## Rate Limiting

Rate limits are enforced per API key:

| Plan | Requests/min | Tokens/month |
|------|-------------|--------------|
| FREE | 10 | 10,000 |
| PRO | 100 | 100,000 |
| ENTERPRISE | 1,000 | 1,000,000 |

When rate limit is exceeded:
```json
{
  "detail": "Rate limit exceeded"
}
```
**HTTP Status**: 429

## Error Handling

### Authentication Errors (401)
```json
{
  "detail": "Missing or invalid Authorization header"
}
```

### Rate Limit Errors (429)
```json
{
  "detail": "Rate limit exceeded"
}
```

### Server Errors (500)
```json
{
  "detail": "Internal server error message"
}
```

## Local Testing

### 1. Start the FastAPI server

```bash
# From project root
cd orchestrator
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test with curl

```bash
# Test chat endpoint
curl -X POST http://localhost:8000/v1/chat \
  -H "Authorization: ApiKey am_live_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, AetherMind!",
    "namespace": "universal"
  }'
```

### 3. Test with Python SDK

```python
from aethermind import AetherMindClient

client = AetherMindClient(
    api_key="am_live_xxx",
    base_url="http://localhost:8000"  # For local testing
)

response = client.chat(message="What is AI?")
print(response['answer'])
```

### 4. Run comprehensive tests

```bash
export AETHERMIND_API_KEY=am_live_xxx
python sdk/test_api_endpoints.py
```

## Production Deployment

### Deploy to Render

1. **Create new Web Service** on Render.com
2. **Connect GitHub repository**
3. **Configure build settings**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd orchestrator && uvicorn main_api:app --host 0.0.0.0 --port $PORT`
4. **Set environment variables** (all from `.env` file)
5. **Deploy**

### Deploy to Railway

1. **Create new project** on Railway.app
2. **Connect GitHub repository**
3. **Add environment variables**
4. **Set start command**: `cd orchestrator && uvicorn main_api:app --host 0.0.0.0 --port $PORT`
5. **Deploy**

### Deploy to AWS EC2

```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Clone repository
git clone https://github.com/yourusername/aethermind_universal.git
cd aethermind_universal

# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set environment variables
nano .env  # Add all variables

# Install process manager
pip install gunicorn

# Start with Gunicorn
cd orchestrator
gunicorn main_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### NGINX Configuration

```nginx
server {
    listen 80;
    server_name api.aethermind.ai;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring

### Health Check Endpoint

```bash
curl http://localhost:8000/health
```

### Logs

```bash
# View logs in production
tail -f logs/aethermind.log

# Or with journalctl (systemd)
journalctl -u aethermind -f
```

### Metrics

The backend logs:
- Request count
- Response times
- Error rates
- Token usage
- Rate limit hits

## Security Best Practices

1. **Always use HTTPS** in production
2. **Rotate API keys** regularly
3. **Monitor for unusual activity**
4. **Keep dependencies updated**
5. **Use environment variables** for secrets
6. **Enable CORS** only for trusted domains
7. **Implement audit logging** for all requests

## Troubleshooting

### Issue: "Invalid API key"
**Solution**: Verify the key format starts with `am_live_` or `am_test_`

### Issue: "Rate limit exceeded"
**Solution**: Check usage with `/v1/usage` endpoint, upgrade plan if needed

### Issue: "Perception Service unavailable"
**Solution**: Verify `PERCEPTION_SERVICE_URL` is set and service is running

### Issue: "AETHER loop timeout"
**Solution**: Check RunPod service status, verify `RUNPOD_API_KEY`

## API Versioning

All endpoints are versioned with `/v1/` prefix. Future versions will use:
- `/v2/` - For breaking changes
- `/v1.1/` - For minor updates (optional)

## Support

- **Documentation**: https://aethermind.ai/documentation
- **GitHub Issues**: https://github.com/yourusername/aethermind_universal/issues
- **Email**: support@aethermind.ai

---

**Last Updated**: January 2024
**API Version**: 1.0.0
