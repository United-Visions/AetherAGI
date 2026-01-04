/**
 * AetherMind JavaScript/TypeScript SDK
 * Official client for AetherMind AGI API
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

export interface AetherMindConfig {
  apiKey: string;
  baseURL?: string;
  timeout?: number;
}

export interface ChatOptions {
  message: string;
  namespace?: string;
  stream?: boolean;
  maxTokens?: number;
  temperature?: number;
  includeMemory?: boolean;
}

export interface ChatResponse {
  answer: string;
  reasoning_steps?: string[];
  confidence?: number;
  sources?: string[];
  tokens_used?: number;
}

export interface MemorySearchOptions {
  query: string;
  namespace?: string;
  topK?: number;
  includeEpisodic?: boolean;
  includeKnowledge?: boolean;
}

export interface MemoryResult {
  text: string;
  score: number;
  timestamp: string;
  namespace: string;
  metadata?: Record<string, any>;
}

export class AetherMindError extends Error {
  constructor(message: string, public statusCode?: number) {
    super(message);
    this.name = 'AetherMindError';
  }
}

export class AuthenticationError extends AetherMindError {
  constructor(message: string = 'Invalid API key') {
    super(message, 401);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends AetherMindError {
  constructor(message: string = 'Rate limit exceeded') {
    super(message, 429);
    this.name = 'RateLimitError';
  }
}

/**
 * AetherMind Client
 * 
 * @example
 * ```typescript
 * const client = new AetherMindClient({ apiKey: 'am_live_your_key' });
 * const response = await client.chat({ message: 'Hello, AetherMind!' });
 * console.log(response.answer);
 * ```
 */
export class AetherMindClient {
  private client: AxiosInstance;

  constructor(config: AetherMindConfig) {
    if (!config.apiKey) {
      throw new AuthenticationError('API key is required');
    }

    this.client = axios.create({
      baseURL: config.baseURL || process.env.AETHERMIND_BASE_URL || 'https://api.aethermind.ai',
      timeout: config.timeout || 30000,
      headers: {
        'Authorization': `ApiKey ${config.apiKey}`,
        'Content-Type': 'application/json',
        'User-Agent': '@aethermind/sdk/1.0.0'
      }
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      response => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          throw new AuthenticationError();
        } else if (error.response?.status === 429) {
          throw new RateLimitError();
        } else {
          throw new AetherMindError(
            error.message || 'API request failed',
            error.response?.status
          );
        }
      }
    );
  }

  /**
   * Send a chat message to AetherMind
   */
  async chat(options: ChatOptions): Promise<ChatResponse> {
    const response = await this.client.post<ChatResponse>('/v1/chat', {
      message: options.message,
      namespace: options.namespace || 'universal',
      stream: options.stream || false,
      max_tokens: options.maxTokens,
      temperature: options.temperature || 0.7,
      include_memory: options.includeMemory !== false
    });

    return response.data;
  }

  /**
   * Search AetherMind's infinite memory
   */
  async searchMemory(options: MemorySearchOptions): Promise<MemoryResult[]> {
    const response = await this.client.post<{ results: MemoryResult[] }>(
      '/v1/memory/search',
      {
        query: options.query,
        namespace: options.namespace || 'universal',
        top_k: options.topK || 10,
        include_episodic: options.includeEpisodic !== false,
        include_knowledge: options.includeKnowledge !== false
      }
    );

    return response.data.results;
  }

  /**
   * Create a custom tool (ToolForge)
   */
  async createTool(params: {
    name: string;
    description: string;
    code: string;
    parameters: Record<string, any>;
  }): Promise<{ tool_id: string; status: string }> {
    const response = await this.client.post('/v1/tools/create', params);
    return response.data;
  }

  /**
   * Get current usage statistics
   */
  async getUsage(): Promise<{
    requests_remaining: number;
    reset_at: string;
    total_tokens: number;
    plan: string;
  }> {
    const response = await this.client.get('/v1/usage');
    return response.data;
  }

  /**
   * List available knowledge namespaces
   */
  async listNamespaces(): Promise<string[]> {
    const response = await this.client.get<{ namespaces: string[] }>('/v1/namespaces');
    return response.data.namespaces;
  }

  /**
   * Create a knowledge cartridge
   */
  async createKnowledgeCartridge(params: {
    name: string;
    namespace: string;
    documents: string[];
    metadata?: Record<string, any>;
  }): Promise<{ cartridge_id: string; status: string }> {
    const response = await this.client.post('/v1/knowledge/cartridge', params);
    return response.data;
  }
}

// Export all types and classes
export default AetherMindClient;
