// api.js - Handles communication with the backend

console.log('📡 [API] API module loaded');

// API Key Modal will be set by router.js
let apiKeyModal = null;

export function setApiKeyModal(modal) {
    console.log('🔑 [API] Setting API key modal reference');
    apiKeyModal = modal;
}

function handleAuthFailure(status) {
    if (status === 401 || status === 403) {
        console.warn(`⚠️ [API] Auth failure (${status}). Clearing cache and redirecting to reset.`);
        // Redirect to logout endpoint which clears session and localStorage
        window.location.href = "/logout";
    }
}

export const api = {
    // Use Flask proxy for local dev, direct for production
    // Flask proxy handles forwarding to backend on port 8000
    rootUrl: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? ''  // Empty = use same origin (Flask proxy on 5000)
        : 'https://aetheragi.onrender.com',

    get baseUrl() {
        return `${this.rootUrl}/v1/chat/completions`;
    },

    uploadUrl: '/v1/ingest/multimodal', // Relative path to Flask proxy

    getApiKey() {
        console.log('🔑 [API] Getting API key...');
        
        // First, check if API key is in URL query parameter
        const urlParams = new URLSearchParams(window.location.search);
        const urlApiKey = urlParams.get('api_key');
        
        if (urlApiKey) {
            console.log('✅ [API] Found API key in URL parameter');
            localStorage.setItem('aethermind_api_key', urlApiKey);
            return urlApiKey;
        }
        
        // Otherwise, check localStorage
        let key = localStorage.getItem('aethermind_api_key');
        if (!key) {
            console.warn('⚠️ [API] No API key found in localStorage');
            
            // Show modal if available
            if (apiKeyModal) {
                console.log('🔑 [API] Showing API key modal...');
                apiKeyModal.show();
                // Return null and let modal handle submission
                return null;
            } else {
                console.warn('⚠️ [API] Modal not available, falling back to prompt');
                key = prompt("Please enter your AetherMind API Key (AM_LIVE_KEY):");
                if (key) {
                    localStorage.setItem('aethermind_api_key', key);
                    console.log('✅ [API] API key saved to localStorage');
                } else {
                    console.error('❌ [API] No API key provided by user');
                }
            }
        } else {
            console.log('✅ [API] API key found in localStorage');
        }
        return key;
    },

    async sendMessage(messages) {
        console.log('📤 [API] sendMessage called');
        console.log('📝 [API] Messages:', messages);
        console.log('🌐 [API] Target URL:', this.baseUrl);
        
        const apiKey = this.getApiKey();
        if (!apiKey) {
            console.error('❌ [API] No API key available');
            throw new Error("API Key required");
        }

        const payload = {
            model: 'aethermind-v1',
            user: 'flask_user_01',
            messages: messages,
        };
        console.log('📦 [API] Request payload:', payload);

        try {
            console.log('⏳ [API] Sending request...');
            const startTime = performance.now();
            
            const response = await fetch(this.baseUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': apiKey
                },
                body: JSON.stringify(payload),
            });
            
            const endTime = performance.now();
            console.log(`⏱️ [API] Request completed in ${(endTime - startTime).toFixed(2)}ms`);
            console.log('📊 [API] Response status:', response.status, response.statusText);

            if (!response.ok) {
                handleAuthFailure(response.status);
                const errorText = await response.text();
                console.error('❌ [API] Request failed:', response.status, errorText);
                throw new Error(`API Error: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            console.log('✅ [API] Response data:', data);
            return data;
        } catch (error) {
            console.error('❌ [API] sendMessage error:', error);
            console.error('❌ [API] Error stack:', error.stack);
            throw error;
        }
    },

    // Chat with context support (for onboarding, special modes)
    async chat(options) {
        console.log('💬 [API] chat called with context:', options.context?.mode);
        
        const apiKey = this.getApiKey();
        if (!apiKey) {
            console.error('❌ [API] No API key available');
            throw new Error("API Key required");
        }

        const payload = {
            model: 'aethermind-v1',
            user: 'flask_user_01',
            messages: options.messages || [],
            context: options.context || {}
        };

        try {
            const response = await fetch(this.baseUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': apiKey
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                handleAuthFailure(response.status);
                const errorText = await response.text();
                console.error('❌ [API] Chat failed:', response.status, errorText);
                throw new Error(`API Error: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            console.log('✅ [API] Chat response:', data);
            return data;
        } catch (error) {
            console.error('❌ [API] chat error:', error);
            throw error;
        }
    },

    async uploadFile(file) {
        console.log('📤 [API] uploadFile called');
        console.log('📄 [API] File:', file.name, 'Size:', file.size, 'Type:', file.type);
        console.log('🌐 [API] Upload URL:', this.uploadUrl);
        
        const apiKey = this.getApiKey();
        if (!apiKey) {
            console.error('❌ [API] No API key available for upload');
            throw new Error("API Key required for uploads");
        }

        const formData = new FormData();
        formData.append('file', file);
        console.log('📦 [API] FormData prepared');

        try {
            console.log('⏳ [API] Uploading file...');
            const startTime = performance.now();
            
            const response = await fetch(this.uploadUrl, {
                method: 'POST',
                headers: {
                    'Aether-Secret-Key': apiKey,
                },
                body: formData,
            });
            
            const endTime = performance.now();
            console.log(`⏱️ [API] Upload completed in ${(endTime - startTime).toFixed(2)}ms`);
            console.log('📊 [API] Upload response status:', response.status, response.statusText);

            if (!response.ok) {
                if (response.status === 503) {
                    console.warn('⚠️ [API] Service unavailable (503) - Model warming up');
                    throw new Error("Perception model warming up, please try again.");
                }
                handleAuthFailure(response.status);
                const errorText = await response.text();
                console.error('❌ [API] Upload failed:', response.status, errorText);
                throw new Error(`Upload Error: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            console.log('✅ [API] Upload response data:', data);
            return data;
        } catch (error) {
            console.error('❌ [API] uploadFile error:', error);
            console.error('❌ [API] Error stack:', error.stack);
            throw error;
        }
    },

    // ============================================================================
    // NEW SHELL UI ENDPOINTS
    // ============================================================================

    async getUserProfile() {
        console.log('👤 [API] getUserProfile called');
        
        const apiKey = this.getApiKey();
        if (!apiKey) return null;

        try {
            const response = await fetch(`${this.rootUrl}/v1/user/profile`, {
                method: 'GET',
                headers: {
                    'X-Aether-Key': apiKey
                }
            });

            if (!response.ok) {
                handleAuthFailure(response.status);
                console.warn('⚠️ [API] Failed to get user profile');
                return null;
            }

            const data = await response.json();
            console.log('✅ [API] User profile:', data);
            return data;
        } catch (error) {
            console.error('❌ [API] getUserProfile error:', error);
            return null;
        }
    },

    async saveUserProfile(profile) {
        console.log('💾 [API] saveUserProfile called');
        
        const apiKey = this.getApiKey();
        if (!apiKey) {
            console.warn('⚠️ [API] synthesizeVoice aborted: missing API key');
            if (apiKeyModal) {
                apiKeyModal.show();
            }
            throw new Error("API Key required for voice synthesis");
        }

        try {
            const response = await fetch(`${this.rootUrl}/v1/user/profile`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': apiKey
                },
                body: JSON.stringify(profile)
            });

            if (!response.ok) {
                handleAuthFailure(response.status);
                throw new Error(`Failed to save profile: ${response.status}`);
            }

            const data = await response.json();
            console.log('✅ [API] Profile saved:', data);
            return data;
        } catch (error) {
            console.error('❌ [API] saveUserProfile error:', error);
            throw error;
        }
    },

    async getTasksStatus(taskIds) {
        console.log('📋 [API] getTasksStatus called');
        
        const apiKey = this.getApiKey();
        if (!apiKey) return { tasks: [] };

        try {
            const response = await fetch(`${this.rootUrl}/v1/tasks/status`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': apiKey
                },
                body: JSON.stringify({ task_ids: taskIds })
            });

            if (!response.ok) {
                console.warn('⚠️ [API] Failed to get task status');
                return { tasks: [] };
            }

            return await response.json();
        } catch (error) {
            console.error('❌ [API] getTasksStatus error:', error);
            return { tasks: [] };
        }
    },

    async createTask(taskType, taskName, taskData = {}) {
        console.log('➕ [API] createTask called');
        
        const apiKey = this.getApiKey();
        if (!apiKey) {
            console.warn('⚠️ [API] synthesizeVoice aborted: missing API key');
            if (apiKeyModal) {
                apiKeyModal.show();
            }
            throw new Error("API Key required");
        }

        try {
            const response = await fetch(`${this.rootUrl}/v1/tasks/create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': apiKey
                },
                body: JSON.stringify({
                    task_type: taskType,
                    task_name: taskName,
                    task_data: taskData
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to create task: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('❌ [API] createTask error:', error);
            throw error;
        }
    },

    async runBenchmark(benchmarkType) {
        console.log('🏃 [API] runBenchmark called:', benchmarkType);
        
        const apiKey = this.getApiKey();
        if (!apiKey) throw new Error("API Key required");

        try {
            const response = await fetch(`${this.rootUrl}/v1/benchmarks/run?family=${benchmarkType}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': apiKey
                }
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to run benchmark: ${response.status} - ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('❌ [API] runBenchmark error:', error);
            throw error;
        }
    },

    async forgeTool(spec) {
        console.log('🔧 [API] forgeTool called:', spec.name);
        
        const apiKey = this.getApiKey();
        if (!apiKey) throw new Error("API Key required");

        try {
            const response = await fetch(`${this.rootUrl}/v1/tools/create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': apiKey
                },
                body: JSON.stringify(spec)
            });

            if (!response.ok) {
                throw new Error(`Failed to forge tool: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('❌ [API] forgeTool error:', error);
            throw error;
        }
    },

    // ============================================================================
    // PERSONA MANAGEMENT
    // ============================================================================

    async createPersona(name, description, traits = [], speechStyle = 'Natural and conversational') {
        console.log('🎭 [API] createPersona:', name);
        
        const apiKey = this.getApiKey();
        if (!apiKey) throw new Error("API Key required");

        try {
            const response = await fetch(`${this.rootUrl}/v1/personas/create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': apiKey
                },
                body: JSON.stringify({
                    name,
                    description,
                    traits,
                    speech_style: speechStyle
                })
            });

            if (!response.ok) throw new Error(`Failed to create persona: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('❌ [API] createPersona error:', error);
            throw error;
        }
    },

    async listPersonas() {
        console.log('🎭 [API] listPersonas');
        
        const apiKey = this.getApiKey();
        if (!apiKey) return { personas: {}, active_persona: null };

        try {
            const response = await fetch(`${this.rootUrl}/v1/personas`, {
                headers: { 'X-Aether-Key': apiKey }
            });

            if (!response.ok) return { personas: {}, active_persona: null };
            return await response.json();
        } catch (error) {
            console.error('❌ [API] listPersonas error:', error);
            return { personas: {}, active_persona: null };
        }
    },

    async switchPersona(personaName) {
        console.log('🎭 [API] switchPersona:', personaName);
        
        const apiKey = this.getApiKey();
        if (!apiKey) throw new Error("API Key required");

        try {
            const response = await fetch(`${this.rootUrl}/v1/personas/switch/${encodeURIComponent(personaName)}`, {
                method: 'POST',
                headers: { 'X-Aether-Key': apiKey }
            });

            if (!response.ok) throw new Error(`Failed to switch persona: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('❌ [API] switchPersona error:', error);
            throw error;
        }
    },

    async deletePersona(personaName) {
        console.log('🎭 [API] deletePersona:', personaName);
        
        const apiKey = this.getApiKey();
        if (!apiKey) throw new Error("API Key required");

        try {
            const response = await fetch(`${this.rootUrl}/v1/personas/${encodeURIComponent(personaName)}`, {
                method: 'DELETE',
                headers: { 'X-Aether-Key': apiKey }
            });

            if (!response.ok) throw new Error(`Failed to delete persona: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('❌ [API] deletePersona error:', error);
            throw error;
        }
    },

    // ============================================================================
    // VOICE SYNTHESIS (Edge TTS)
    // ============================================================================

    async synthesizeVoice(text, options = {}) {
        console.log('🔊 [API] synthesizeVoice called');
        
        const apiKey = this.getApiKey();
        if (!apiKey) throw new Error("API Key required");

        try {
            const response = await fetch(`${this.rootUrl}/v1/voice/synthesize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': apiKey
                },
                body: JSON.stringify({
                    text: text,
                    voice_id: options.voiceId || null,
                    persona: options.persona || null,
                    rate: options.rate || '+0%',
                    pitch: options.pitch || '+0Hz'
                })
            });

            if (!response.ok) {
                handleAuthFailure(response.status);
                throw new Error(`Voice synthesis failed: ${response.status}`);
            }

            const data = await response.json();
            console.log('✅ [API] Voice synthesized:', data.voice_used);
            return data;
        } catch (error) {
            console.error('❌ [API] synthesizeVoice error:', error);
            throw error;
        }
    },

    async listVoices(language = 'en') {
        console.log('🔊 [API] listVoices called');
        
        const apiKey = this.getApiKey();
        if (!apiKey) {
            console.warn('⚠️ [API] listVoices aborted: missing API key');
            if (apiKeyModal) {
                apiKeyModal.show();
            }
            return { voices: [], profiles: {} };
        }

        try {
            const response = await fetch(`${this.rootUrl}/v1/voice/voices?language=${language}`, {
                headers: { 'X-Aether-Key': apiKey }
            });

            if (!response.ok) return { voices: [], profiles: {} };
            return await response.json();
        } catch (error) {
            console.error('❌ [API] listVoices error:', error);
            return { voices: [], profiles: {} };
        }
    }
};

console.log('✅ [API] API module initialized with baseUrl:', api.baseUrl);
