# AetherMind API Authentication Guide

**Last Updated:** January 4, 2026  
**Purpose:** Complete guide for integrating AetherMind into any application

---

## 🔑 Authentication Methods

AetherMind supports **three authentication methods**:

1. **API Keys** - Best for server-to-server, long-lived credentials
2. **JWT Tokens** - Best for web/mobile apps, short-lived, stateless
3. **Sessions** - Best for traditional web apps with cookies

---

## 1. API Key Authentication

### Generate API Key

```python
from orchestrator.auth_manager import AuthManager, UserRole

auth = AuthManager()

# Generate key for a user
api_key = auth.generate_api_key(
    user_id="user_github123",
    role=UserRole.PRO,
    metadata={"email": "user@example.com", "name": "John Doe"}
)

print(f"Your API key: {api_key}")
# Output: am_live_XXXXXXXXXXXXXXXXXXXXX
```

### Use API Key

**Format:** `Authorization: ApiKey am_live_XXXXX`

```bash
curl -X POST https://api.aethermind.ai/v1/chat \
  -H "Authorization: ApiKey am_live_YOUR_KEY_HERE" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello AetherMind"}'
```

### Verify API Key

```python
# Server-side verification
user_data = auth.verify_api_key("am_live_XXXXX")

if user_data:
    print(f"User: {user_data['user_id']}")
    print(f"Role: {user_data['role']}")
    print(f"Permissions: {user_data['permissions']}")
else:
    print("Invalid API key")
```

---

## 2. JWT Token Authentication

### Generate JWT Token

```python
from orchestrator.auth_manager import AuthManager, UserRole

auth = AuthManager()

# Generate JWT token (expires in 1 hour)
jwt_token = auth.generate_jwt_token(
    user_id="user_github123",
    role=UserRole.PRO,
    expires_in=3600  # 1 hour in seconds
)

print(f"Your JWT: {jwt_token}")
```

### Use JWT Token

**Format:** `Authorization: Bearer <jwt_token>`

```bash
curl -X POST https://api.aethermind.ai/v1/chat \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello AetherMind"}'
```

### Verify JWT Token

```python
# Server-side verification
user_data = auth.verify_jwt_token(jwt_token)

if user_data:
    print(f"User: {user_data['user_id']}")
    print(f"Role: {user_data['role']}")
else:
    print("Invalid or expired token")
```

---

## 3. Session Authentication

### Create Session

```python
# After user login
session_id = auth.create_session(
    user_id="user_github123",
    role=UserRole.PRO,
    duration=86400  # 24 hours
)

# Store session_id in cookie or return to client
```

### Verify Session

```python
# On each request
user_data = auth.verify_session(session_id)

if user_data:
    # User is authenticated
    pass
else:
    # Session expired or invalid
    pass
```

---

## 📦 Platform Integration Examples

### Python

```python
# Install: pip install requests

import requests

class AetherMindClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.aethermind.ai/v1"
    
    def chat(self, message):
        response = requests.post(
            f"{self.base_url}/chat",
            headers={"Authorization": f"ApiKey {self.api_key}"},
            json={"message": message}
        )
        return response.json()

# Usage
client = AetherMindClient("am_live_YOUR_KEY")
result = client.chat("Explain quantum computing")
print(result)
```

### Next.js (TypeScript)

```typescript
// lib/aethermind.ts
export class AetherMindClient {
  private apiKey: string;
  private baseUrl = 'https://api.aethermind.ai/v1';

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async chat(message: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Authorization': `ApiKey ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message })
    });

    return response.json();
  }
}

// app/api/chat/route.ts (Server-side API route)
import { AetherMindClient } from '@/lib/aethermind';

export async function POST(request: Request) {
  const client = new AetherMindClient(process.env.AETHERMIND_API_KEY!);
  const { message } = await request.json();
  
  const result = await client.chat(message);
  return Response.json(result);
}

// app/components/Chat.tsx (Client component)
'use client';

export default function Chat() {
  const [message, setMessage] = useState('');
  
  const sendMessage = async () => {
    const response = await fetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message })
    });
    
    const result = await response.json();
    console.log(result);
  };
  
  return (
    <div>
      <input value={message} onChange={(e) => setMessage(e.target.value)} />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}
```

### React (JavaScript)

```javascript
// src/lib/aethermind.js
export class AetherMindClient {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.baseUrl = 'https://api.aethermind.ai/v1';
  }

  async chat(message) {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Authorization': `ApiKey ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message })
    });

    return response.json();
  }
}

// src/components/Chat.jsx
import { useState } from 'react';
import { AetherMindClient } from '../lib/aethermind';

export default function Chat() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  
  const client = new AetherMindClient(import.meta.env.VITE_AETHERMIND_API_KEY);
  
  const sendMessage = async () => {
    const result = await client.chat(message);
    setResponse(result.response);
  };
  
  return (
    <div>
      <input value={message} onChange={(e) => setMessage(e.target.value)} />
      <button onClick={sendMessage}>Send</button>
      <div>{response}</div>
    </div>
  );
}
```

### Node.js/Express

```javascript
// server.js
const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

const AETHERMIND_API_KEY = process.env.AETHERMIND_API_KEY;

app.post('/api/chat', async (req, res) => {
  try {
    const response = await axios.post(
      'https://api.aethermind.ai/v1/chat',
      { message: req.body.message },
      {
        headers: {
          'Authorization': `ApiKey ${AETHERMIND_API_KEY}`
        }
      }
    );
    
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => console.log('Server running on port 3000'));
```

### React Native (Mobile)

```javascript
// services/aethermind.js
export class AetherMindClient {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.baseUrl = 'https://api.aethermind.ai/v1';
  }

  async chat(message) {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Authorization': `ApiKey ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message })
    });

    return response.json();
  }
}

// screens/ChatScreen.js
import React, { useState } from 'react';
import { View, TextInput, Button, Text } from 'react-native';
import { AetherMindClient } from '../services/aethermind';

export default function ChatScreen() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  
  const client = new AetherMindClient('am_live_YOUR_KEY');
  
  const sendMessage = async () => {
    const result = await client.chat(message);
    setResponse(result.response);
  };
  
  return (
    <View>
      <TextInput value={message} onChangeText={setMessage} />
      <Button title="Send" onPress={sendMessage} />
      <Text>{response}</Text>
    </View>
  );
}
```

### Flutter (Dart)

```dart
// lib/services/aethermind_client.dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class AetherMindClient {
  final String apiKey;
  final String baseUrl = 'https://api.aethermind.ai/v1';

  AetherMindClient(this.apiKey);

  Future<Map<String, dynamic>> chat(String message) async {
    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      headers: {
        'Authorization': 'ApiKey $apiKey',
        'Content-Type': 'application/json'
      },
      body: jsonEncode({'message': message})
    );

    return jsonDecode(response.body);
  }
}

// lib/screens/chat_screen.dart
import 'package:flutter/material.dart';
import '../services/aethermind_client.dart';

class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final client = AetherMindClient('am_live_YOUR_KEY');
  final controller = TextEditingController();
  String response = '';

  void sendMessage() async {
    final result = await client.chat(controller.text);
    setState(() {
      response = result['response'];
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          TextField(controller: controller),
          ElevatedButton(onPressed: sendMessage, child: Text('Send')),
          Text(response)
        ]
      )
    );
  }
}
```

---

## 🔐 Role-Based Access Control (RBAC)

### Roles & Permissions

```python
from orchestrator.auth_manager import UserRole, Permission

# Available Roles
UserRole.FREE        # Free tier users
UserRole.PRO         # Pro subscribers
UserRole.ENTERPRISE  # Enterprise customers
UserRole.ADMIN       # System administrators

# Available Permissions
Permission.READ              # Read data
Permission.WRITE             # Write data
Permission.DELETE            # Delete data
Permission.META_CONTROLLER   # Use meta-controller
Permission.SELF_MODIFY       # Self-modification
Permission.TOOL_FORGE        # Tool generation
Permission.MANAGE_USERS      # User management
Permission.VIEW_AUDIT        # View audit logs
Permission.MANAGE_KEYS       # Key management
```

### Check Permissions

```python
# Verify user has specific permission
if auth.has_permission(user_data, Permission.TOOL_FORGE):
    # User can use ToolForge
    pass
else:
    return {"error": "Permission denied"}
```

### Upgrade User Role

```python
# Upgrade user from FREE to PRO
auth.upgrade_user_role("user_github123", UserRole.PRO)
```

---

## 📊 Rate Limiting

Rate limits are enforced per role:

| Role | Requests per Minute |
|------|---------------------|
| FREE | 100 |
| PRO | 1,000 |
| ENTERPRISE | 10,000 |
| ADMIN | Unlimited |

```python
# Rate limits are checked automatically
user_data = auth.verify_api_key(api_key)

if user_data is None:
    # Either invalid key OR rate limit exceeded
    return {"error": "Rate limit exceeded"}
```

---

## 🔍 Audit Logging

All authentication events are logged for compliance:

```python
# Get audit logs for a user
logs = auth.get_audit_log(user_id="user_github123", limit=100)

for log in logs:
    print(f"{log['timestamp']}: {log['event']} - {log['success']}")

# Example output:
# 2026-01-04T10:30:00Z: api_key_generated - True
# 2026-01-04T10:31:00Z: api_key_verify - True
# 2026-01-04T10:32:00Z: jwt_generated - True
```

---

## 🛡️ Security Best Practices

### 1. Store Keys Securely

**❌ DON'T:**
```javascript
const apiKey = 'am_live_abc123';  // Hardcoded in client
```

**✅ DO:**
```javascript
// In .env file (server-side)
AETHERMIND_API_KEY=am_live_abc123

// In code (server-side only)
const apiKey = process.env.AETHERMIND_API_KEY;
```

### 2. Never Expose Keys in Frontend

**❌ DON'T:**
```typescript
// ❌ Client-side (exposed to browser)
const response = await fetch('https://api.aethermind.ai/v1/chat', {
  headers: { 'Authorization': `ApiKey ${process.env.NEXT_PUBLIC_KEY}` }
});
```

**✅ DO:**
```typescript
// ✅ Call your own backend API
const response = await fetch('/api/chat', {
  method: 'POST',
  body: JSON.stringify({ message })
});

// Backend (app/api/chat/route.ts) handles auth
export async function POST(request: Request) {
  // API key only exists on server
  const response = await fetch('https://api.aethermind.ai/v1/chat', {
    headers: { 'Authorization': `ApiKey ${process.env.AETHERMIND_API_KEY}` }
  });
  return response;
}
```

### 3. Use JWT for Client Apps

For web/mobile apps, use JWT tokens with short expiration:

```python
# Generate short-lived token (15 minutes)
token = auth.generate_jwt_token(user_id, role, expires_in=900)
```

### 4. Rotate Keys Regularly

```python
# Revoke old key
auth.revoke_api_key(old_api_key)

# Generate new key
new_api_key = auth.generate_api_key(user_id, role)
```

---

## 🆘 Error Handling

```python
user_data = auth.verify_request(auth_header)

if user_data is None:
    # Possible reasons:
    # 1. Invalid API key/token
    # 2. Expired JWT token
    # 3. Rate limit exceeded
    # 4. Revoked key
    
    return {
        "error": "Authentication failed",
        "code": 401
    }
```

---

## 📚 Additional Resources

- [API Reference](./API_REFERENCE.md)
- [SDK Documentation](./SDK_DOCUMENTATION.md)
- [Migration Guide](./MIGRATION_GUIDE.md)
- [Security Best Practices](./SECURITY.md)

---

## 🤝 Support

- **Email:** support@aethermind.ai
- **Discord:** discord.gg/aethermind
- **GitHub Issues:** github.com/united-visions/aethermind/issues
