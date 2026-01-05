# AetherMind API Key Architecture

## Overview

AetherMind uses a **two-tier API key system** to separate backend service authentication from user authentication.

---

## 🔑 Two Types of Keys

### 1. **Backend Service Keys** (Stored in `.env`)
These authenticate **AetherMind's backend** with external services.

| Key Name | Purpose | Where Used |
|----------|---------|------------|
| `PINECONE_API_KEY` | Vector database access | Orchestrator → Pinecone |
| `RUNPOD_API_KEY` | GPU inference | Orchestrator → RunPod |
| `GEMINI_API_KEY` | LLM inference (backup) | Brain → Gemini |
| `FIRECRAWL_API_KEY` | Web scraping | Mind → FireCrawl |
| `REDIS_URL` | Job scheduling | Curiosity → Redis |

**Storage:** `.env` file (never exposed to users)  
**Security:** Server-side only, deployed to Render as secrets

---

### 2. **User Personal Keys** (Format: `am_live_XXXXX`)
These authenticate **individual users** with AetherMind's API.

**Key Properties:**
- Format: `am_live_` + 32-character random string
- Generated once per user
- Shown to user **only once** (like GitHub tokens)
- Stored **hashed** (SHA-256) in `config/keys.json`
- Tied to user's GitHub account

**Storage:** `config/keys.json` (encrypted with Fernet)

---

## 🔐 User Key Generation Flow

```
1. User logs in via GitHub OAuth
   ↓
2. User clicks "Create API Key" on onboarding
   ↓
3. Flask app calls: auth_mgr.generate_api_key(user_id, role=UserRole.PRO)
   ↓
4. AuthManager:
   - Generates: am_live_ABC123XYZ789...
   - Hashes with SHA-256
   - Stores hash in keys.json with user_id + role
   ↓
5. Returns plain key to user (ONLY TIME IT'S VISIBLE)
   ↓
6. User copies key and saves it securely
```

---

## 🔒 User Key Validation Flow

```
1. User makes API request with key in header:
   X-Aether-Key: am_live_ABC123XYZ789...
   ↓
2. Backend extracts key from header
   ↓
3. AuthManager:
   - Hashes provided key with SHA-256
   - Looks up hash in keys.json
   - Retrieves user_id + role + permissions
   ↓
4. Backend validates permissions and processes request
```

---

## 👥 User Roles & Permissions

| Role | Rate Limit | Permissions |
|------|------------|-------------|
| **FREE** | 100 req/min | READ, WRITE |
| **PRO** | 1000 req/min | READ, WRITE, DELETE, META_CONTROLLER, SELF_MODIFY, TOOL_FORGE |
| **ENTERPRISE** | 10,000 req/min | All PRO + VIEW_AUDIT, MANAGE_KEYS |
| **ADMIN** | Unlimited | All permissions |

---

## 📦 Storage Details

### `config/keys.json` Structure (Encrypted)
```json
{
  "encrypted": true,
  "data": "<Fernet-encrypted JSON>"
}
```

### Decrypted Contents:
```json
{
  "abc123hash...": {
    "user_id": "github_username",
    "role": "pro",
    "created_at": "2026-01-04T12:00:00",
    "last_used": "2026-01-04T14:30:00",
    "metadata": {},
    "revoked": false
  }
}
```

---

## 🚀 Production Migration (TODO)

**Current:** `config/keys.json` (local file)  
**Target:** Supabase PostgreSQL

### Migration Plan:
1. Create `api_keys` table in Supabase
2. Update `AuthManager` to use Supabase client
3. Migrate existing keys from JSON to Supabase
4. Deploy with `SUPABASE_URL` and `SUPABASE_ANON_KEY`

### Benefits:
- ✅ Multi-server access (Render can connect)
- ✅ Automatic backups
- ✅ Row-level security
- ✅ Real-time subscriptions
- ✅ Better scalability

---

## 🔍 Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/create_key` | POST | Generate new user API key |
| `/v1/user/domain` | POST | Set user's domain preference |
| `/v1/user/permissions` | GET | Check user's role & permissions |
| `/v1/chat/completions` | POST | Send chat messages (requires X-Aether-Key) |

---

## ⚠️ Security Best Practices

1. **Never commit `.env` to git** (already in `.gitignore`)
2. **User keys are hashed** - plain text never stored
3. **Keys shown once** - users must save them
4. **Rate limiting** - prevents abuse by role
5. **Encryption at rest** - Fernet encryption for keys.json
6. **Audit logging** - All auth events logged to `config/audit.jsonl`

---

## 🧪 Testing Your Key

```bash
# Test your personal API key
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Aether-Key: am_live_YOUR_KEY_HERE" \
  -d '{
    "model": "aethermind-v1",
    "user": "test_user",
    "messages": [
      {"role": "user", "content": "Hello AetherMind!"}
    ]
  }'
```

```bash
# Check your permissions
curl -X GET http://127.0.0.1:8000/v1/user/permissions \
  -H "X-Aether-Key: am_live_YOUR_KEY_HERE"
```

---

## 📝 Summary

- **Backend service keys** = AetherMind talks to external APIs (Pinecone, RunPod, etc.)
- **User personal keys** = Users talk to AetherMind (am_live_XXX)
- Each user gets their own key with role-based permissions (FREE/PRO/ENTERPRISE)
- Keys are hashed and stored securely in `config/keys.json` (future: Supabase)
- Rate limiting and permissions enforce usage limits per role
