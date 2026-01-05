// api.js - Handles communication with the backend

console.log('📡 [API] API module loaded');

// API Key Modal will be set by router.js
let apiKeyModal = null;

export function setApiKeyModal(modal) {
    console.log('🔑 [API] Setting API key modal reference');
    apiKeyModal = modal;
}

export const api = {
    baseUrl: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://127.0.0.1:8000/v1/chat/completions'
        : 'https://aetheragi.onrender.com/v1/chat/completions',

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
    }
};

console.log('✅ [API] API module initialized with baseUrl:', api.baseUrl);
