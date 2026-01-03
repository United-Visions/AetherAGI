// api.js - Handles communication with the backend

export const api = {
    baseUrl: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://127.0.0.1:8000/v1/chat/completions'
        : 'https://aetheragi.onrender.com/v1/chat/completions',

    uploadUrl: '/v1/ingest/multimodal', // Relative path to Flask proxy

    getApiKey() {
        let key = localStorage.getItem('aethermind_api_key');
        if (!key) {
            key = prompt("Please enter your AetherMind API Key (AM_LIVE_KEY):");
            if (key) localStorage.setItem('aethermind_api_key', key);
        }
        return key;
    },

    async sendMessage(messages) {
        const apiKey = this.getApiKey();
        if (!apiKey) throw new Error("API Key required");

        const response = await fetch(this.baseUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Aether-Key': apiKey
            },
            body: JSON.stringify({
                model: 'aethermind-v1',
                user: 'flask_user_01',
                messages: messages,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API Error: ${response.status} - ${errorText}`);
        }

        return await response.json();
    },

    async uploadFile(file) {
        const apiKey = this.getApiKey();
        if (!apiKey) throw new Error("API Key required for uploads");

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(this.uploadUrl, {
            method: 'POST',
            headers: {
                'X-Aether-Key': apiKey,
            },
            body: formData,
        });

        if (!response.ok) {
            if (response.status === 503) {
                throw new Error("Perception model warming up, please try again.");
            }
            const errorText = await response.text();
            throw new Error(`Upload Error: ${response.status} - ${errorText}`);
        }

        return await response.json();
    }
};
