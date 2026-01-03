// Path: frontend_flask/static/script.js

document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatContainer = document.getElementById('chat-container');
    const thoughtVisualizer = document.getElementById('thought-visualizer');
    
    // Multimodal input buttons
    const cameraBtn = document.getElementById('camera-btn');
    const micBtn = document.getElementById('mic-btn');
    const fileBtn = document.getElementById('file-btn');
    const fileInput = document.getElementById('file-input');
    const videoPreview = document.getElementById('video-preview');

    const aethermindApiUrl = 'https://aetheragi.onrender.com/v1/chat/completions'; // Replace if local
    let messages = [];

    // --- Core Functions ---

    const addMessageToUI = (role, content) => {
        const messageDiv = document.createElement('div');
        const isUser = role === 'user';
        
        messageDiv.className = `flex mb-4 ${isUser ? 'justify-end' : 'justify-start'}`;
        
        const bubble = document.createElement('div');
        bubble.className = `px-4 py-2 rounded-lg max-w-lg ${isUser ? 'bg-blue-600 text-white' : 'bg-gray-700'}`;
        bubble.innerText = content;

        messageDiv.appendChild(bubble);
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    };

    const displayThoughts = async (initialThought) => {
        thoughtVisualizer.innerHTML = `<p>> ${initialThought}</p>`;
        await new Promise(resolve => setTimeout(resolve, 150));
        
        const thoughts = [
            "Analyzing input type...",
            "Querying Mind for relevant context (K-12 & Episodic)...",
            "Assembling Mega-Prompt for Brain...",
            "Sending context to Cognitive Core (Llama-3)...",
            "Brain is reasoning...",
            "Receiving response from Brain...",
            "Verifying output with Safety Inhibitor...",
            "Forwarding response to Chat Adapter..."
        ];
        
        for (const thought of thoughts) {
            const p = document.createElement('p');
            p.innerText = `> ${thought}`;
            thoughtVisualizer.appendChild(p);
            thoughtVisualizer.scrollTop = thoughtVisualizer.scrollHeight;
            await new Promise(resolve => setTimeout(resolve, 250));
        }
    };

    const handleFileUpload = async (file) => {
        if (!file) return;

        addMessageToUI('user', `Uploading file: ${file.name}`);
        await displayThoughts(`File received: ${file.name} (${file.type}). Ingesting with Eye...`);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const apiKey = localStorage.getItem('aethermind_api_key');
            if (!apiKey) {
                addMessageToUI('assistant', 'API Key is required for uploads. Please refresh and enter your key.');
                thoughtVisualizer.innerHTML = '<p class="text-red-400">> Cycle aborted: Missing API Key.</p>';
                return;
            }

            const response = await fetch('/v1/ingest/multimodal', { // Relative URL to our Flask backend
                method: 'POST',
                headers: {
                    'X-Aether-Key': apiKey,
                },
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API Error: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            const analysisMessage = `I've analyzed the file "${file.name}".\n\nAnalysis: ${data.analysis}\nSurprise Score: ${data.surprise.toFixed(4)}`;
            addMessageToUI('assistant', analysisMessage);

            if (data.surprise > 0.5) { // Assuming 0.5 is the threshold from the detector
                addMessageToUI('assistant', 'This information is quite novel to me. I have scheduled autonomous research to learn more about it.');
            }

            thoughtVisualizer.innerHTML += '<p class="text-green-400">> Cycle Complete. File analysis response sent.</p>';

        } catch (error) {
            console.error('File upload failed:', error);
            const errorMessage = `Sorry, the file upload failed: ${error.message}`;
            addMessageToUI('assistant', errorMessage);
            thoughtVisualizer.innerHTML = `<p class="text-red-400">> CRITICAL UPLOAD ERROR: ${error.message}</p>`;
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        const userInput = chatInput.value.trim();
        if (!userInput) return;

        addMessageToUI('user', userInput);
        messages.push({ role: 'user', content: userInput });
        chatInput.value = '';

        await displayThoughts("User input received. Initiating active inference cycle...");

        try {
            const apiKey = localStorage.getItem('aethermind_api_key') || prompt("Please enter your AetherMind API Key (AM_LIVE_KEY):");
            if (!apiKey) {
                addMessageToUI('assistant', 'API Key is required. Please refresh and enter your key.');
                thoughtVisualizer.innerHTML = '<p class="text-red-400">> Cycle aborted: Missing API Key.</p>';
                return;
            }
            localStorage.setItem('aethermind_api_key', apiKey);

            const response = await fetch(aethermindApiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Aether-Key': apiKey },
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

            const data = await response.json();
            const assistantMessage = data.choices[0].message;

            addMessageToUI(assistantMessage.role, assistantMessage.content);
            messages.push(assistantMessage);
            thoughtVisualizer.innerHTML += '<p class="text-green-400">> Cycle Complete. Response sent.</p>';

        } catch (error) {
            console.error('Failed to get response from AetherMind:', error);
            const errorMessage = `Sorry, I encountered an error: ${error.message}`;
            addMessageToUI('assistant', errorMessage);
            thoughtVisualizer.innerHTML = `<p class="text-red-400">> CRITICAL ERROR: ${error.message}</p>`;
        }
    };
    
    // --- Event Listeners for Multimodal Input ---

    cameraBtn.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            videoPreview.srcObject = stream;
            videoPreview.classList.remove('hidden');
            videoPreview.play();
            addMessageToUI('system', 'Camera and microphone activated. Awaiting input.');
            // In a full implementation, you'd add logic to capture a photo or record video.
        } catch (err) {
            console.error("Error accessing camera/mic:", err);
            addMessageToUI('system', 'Error: Could not access camera or microphone. Please check permissions.');
        }
    });

    micBtn.addEventListener('click', async () => {
         try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            addMessageToUI('system', 'Microphone activated. Recording...');
            // In a full implementation, you'd use the MediaRecorder API to record audio and send it.
        } catch (err) {
            console.error("Error accessing mic:", err);
            addMessageToUI('system', 'Error: Could not access microphone. Please check permissions.');
        }
    });

    fileBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        handleFileUpload(file);
    });

    chatForm.addEventListener('submit', handleSubmit);
});
