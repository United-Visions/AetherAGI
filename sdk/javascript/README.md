# AetherMind JavaScript/TypeScript SDK

Official JavaScript/TypeScript SDK for **AetherMind AGI**

[![npm version](https://badge.fury.io/js/%40aethermind%2Fsdk.svg)](https://www.npmjs.com/package/@aethermind/sdk)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Installation

```bash
npm install @aethermind/sdk
# or
yarn add @aethermind/sdk
# or
pnpm add @aethermind/sdk
```

## Quick Start

### TypeScript

```typescript
import { AetherMindClient } from '@aethermind/sdk';

const client = new AetherMindClient({
  apiKey: 'am_live_your_key_here'
});

const response = await client.chat({
  message: 'What is Newton\'s Second Law?'
});

console.log(response.answer);
```

### JavaScript (Node.js)

```javascript
const { AetherMindClient } = require('@aethermind/sdk');

const client = new AetherMindClient({
  apiKey: 'am_live_your_key_here'
});

client.chat({ message: 'Hello, AetherMind!' })
  .then(response => console.log(response.answer))
  .catch(error => console.error(error));
```

### Next.js App Router

```typescript
// app/api/chat/route.ts
import { AetherMindClient } from '@aethermind/sdk';
import { NextResponse } from 'next/server';

const client = new AetherMindClient({
  apiKey: process.env.AETHERMIND_API_KEY!
});

export async function POST(request: Request) {
  const { message } = await request.json();
  
  try {
    const response = await client.chat({ message });
    return NextResponse.json(response);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to chat' }, { status: 500 });
  }
}
```

### React Hook

```typescript
// hooks/useAetherMind.ts
import { useState } from 'react';
import { AetherMindClient } from '@aethermind/sdk';

const client = new AetherMindClient({
  apiKey: process.env.NEXT_PUBLIC_AETHERMIND_API_KEY!
});

export function useAetherMind() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const chat = async (message: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await client.chat({ message });
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { chat, loading, error };
}
```

### Express.js

```typescript
import express from 'express';
import { AetherMindClient } from '@aethermind/sdk';

const app = express();
const client = new AetherMindClient({
  apiKey: process.env.AETHERMIND_API_KEY!
});

app.use(express.json());

app.post('/api/chat', async (req, res) => {
  try {
    const { message } = req.body;
    const response = await client.chat({ message });
    res.json(response);
  } catch (error) {
    res.status(500).json({ error: 'Chat failed' });
  }
});

app.listen(3000);
```

## API Reference

### Constructor

```typescript
new AetherMindClient({
  apiKey: string;        // Required: Your API key
  baseURL?: string;      // Optional: Custom API endpoint
  timeout?: number;      // Optional: Request timeout in ms (default: 30000)
});
```

### Methods

#### `chat(options)`

```typescript
await client.chat({
  message: string;           // Required: Your message
  namespace?: string;        // Optional: Knowledge domain (default: 'universal')
  stream?: boolean;          // Optional: Enable streaming (default: false)
  maxTokens?: number;        // Optional: Max response tokens
  temperature?: number;      // Optional: 0.0-1.0 (default: 0.7)
  includeMemory?: boolean;   // Optional: Include past conversations (default: true)
});
```

#### `searchMemory(options)`

```typescript
await client.searchMemory({
  query: string;                  // Required: Search query
  namespace?: string;             // Optional: Domain to search
  topK?: number;                  // Optional: Number of results (default: 10)
  includeEpisodic?: boolean;      // Optional: Include conversations (default: true)
  includeKnowledge?: boolean;     // Optional: Include knowledge (default: true)
});
```

#### `createTool(params)`

```typescript
await client.createTool({
  name: string;                   // Tool name
  description: string;            // What it does
  code: string;                   // JavaScript code
  parameters: object;             // JSON schema
});
```

#### `getUsage()`

```typescript
await client.getUsage();
// Returns: { requests_remaining, reset_at, total_tokens, plan }
```

#### `listNamespaces()`

```typescript
await client.listNamespaces();
// Returns: string[] of available namespaces
```

## Error Handling

```typescript
import { 
  AetherMindError, 
  AuthenticationError, 
  RateLimitError 
} from '@aethermind/sdk';

try {
  const response = await client.chat({ message: 'Hello' });
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error('Invalid API key');
  } else if (error instanceof RateLimitError) {
    console.error('Rate limit exceeded');
  } else if (error instanceof AetherMindError) {
    console.error('API error:', error.message);
  }
}
```

## Environment Variables

Create `.env.local`:

```env
AETHERMIND_API_KEY=am_live_your_key_here
AETHERMIND_BASE_URL=https://api.aethermind.ai
```

## TypeScript Support

Full TypeScript support with type definitions included:

```typescript
import type { 
  ChatResponse, 
  MemoryResult, 
  AetherMindConfig 
} from '@aethermind/sdk';
```

## Publishing to npm

```bash
# Build
npm run build

# Publish
npm publish --access public
```

## License

Apache-2.0
