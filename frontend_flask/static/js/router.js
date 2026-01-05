// router.js - Application Entry Point with Advanced AGI Features

import { api, setApiKeyModal } from './api.js';
import { ChatInterface } from './components/ChatInterface.js';
import { ThinkingVisualizer } from './components/ThinkingVisualizer.js';
import { FileUploader } from './components/FileUploader.js';
import { ActivityFeed } from './components/ActivityFeed.js';
import { SplitViewPanel } from './components/SplitViewPanel.js';
import { BrainVisualizer } from './components/BrainVisualizer.js';
import { ApiKeyModal } from './components/ApiKeyModal.js';

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 [ROUTER] DOMContentLoaded - Initializing AetherMind frontend...');
    
    // Initialize API Key Modal
    console.log('🔑 [ROUTER] Creating ApiKeyModal...');
    const apiKeyModal = new ApiKeyModal();
    setApiKeyModal(apiKeyModal);
    window.apiKeyModal = apiKeyModal; // Make globally accessible
    console.log('✅ [ROUTER] ApiKeyModal initialized');
    
    // Check if API key exists, if not show modal immediately
    const apiKey = localStorage.getItem('aethermind_api_key');
    if (!apiKey) {
        console.warn('⚠️ [ROUTER] No API key found, showing modal...');
        apiKeyModal.show();
    }
    
    // Initialize Core Components
    console.log('📦 [ROUTER] Creating core components...');
    const chat = new ChatInterface('messages');
    console.log('✅ [ROUTER] ChatInterface initialized');
    
    const visualizer = new ThinkingVisualizer('visualizer-container');
    console.log('✅ [ROUTER] ThinkingVisualizer initialized');
    
    const fileUploader = new FileUploader('file-input', 'file-previews', (files) => {
        console.log('📎 [ROUTER] FileUploader callback - files selected:', files.length);
    });
    console.log('✅ [ROUTER] FileUploader initialized');

    // Initialize NEW AGI Components
    console.log('🧠 [ROUTER] Creating AGI components...');
    const activityFeed = new ActivityFeed('activity-feed-container');
    console.log('✅ [ROUTER] ActivityFeed initialized');
    
    const splitView = new SplitViewPanel('split-view-container');
    console.log('✅ [ROUTER] SplitViewPanel initialized');
    
    const brainViz = new BrainVisualizer('brain-visualizer-container');
    console.log('✅ [ROUTER] BrainVisualizer initialized');

    // Make components globally accessible for testing/debugging
    window.activityFeed = activityFeed;
    window.splitView = splitView;
    window.brainViz = brainViz;
    window.chat = chat;

    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const cameraBtn = document.getElementById('camera-btn');
    const micBtn = document.getElementById('mic-btn');
    const fileBtn = document.getElementById('file-btn');
    const chatWrapper = document.getElementById('chat-wrapper');

    // Toggle buttons
    const activityFeedToggle = document.getElementById('activity-feed-toggle');
    const brainVisualizerToggle = document.getElementById('brain-visualizer-toggle');

    let messageHistory = [];
    let activityFeedOpen = false;
    let brainVisualizerOpen = false;

    // Load user domain from localStorage
    const userDomain = localStorage.getItem('aethermind_domain') || 'general';
    console.log('🎯 [ROUTER] User domain loaded:', userDomain);
    updateDomainIndicator(userDomain);
    console.log('✅ [ROUTER] Domain indicator updated');

    // --- Handlers ---

    const handleSend = async (e) => {
        e.preventDefault();
        console.log('📤 [ROUTER] handleSend triggered');
        
        const text = chatInput.value.trim();
        const files = fileUploader.getFiles();
        console.log('📝 [ROUTER] Message text:', text ? `"${text.substring(0, 50)}..."` : '(empty)');
        console.log('📎 [ROUTER] Files attached:', files.length);

        if (!text && files.length === 0) {
            console.warn('⚠️ [ROUTER] No text or files provided, aborting send');
            return;
        }

        // Clear Inputs
        chatInput.value = '';
        fileUploader.clear();

        // 1. Process Files First
        if (files.length > 0) {
            console.log('📂 [ROUTER] Processing files:', files.length);
            for (const file of files) {
                console.log('📄 [ROUTER] Processing file:', file.name, 'Type:', file.type, 'Size:', file.size);
                
                // Activity Feed: Log file upload
                const uploadActivityId = activityFeed.addActivity({
                    id: `upload_${Date.now()}`,
                    type: 'file_change',
                    status: 'in_progress',
                    title: `Uploading ${file.name}`,
                    details: 'Processing multimodal input...',
                    timestamp: new Date().toISOString(),
                    data: { files: [file.name] }
                });

                // UI: Show file sent
                chat.addFileAttachmentMessage('user', file.name, file.type);

                // Visualizer: Start Upload
                const card = visualizer.createCard(`Uploading ${file.name}`);

                try {
                    console.log('⬆️ [ROUTER] Uploading file to API:', file.name);
                    const result = await api.uploadFile(file);
                    console.log('✅ [ROUTER] File upload successful:', file.name, 'Result:', result);
                    
                    card.append(`Analysis: ${result.analysis}`);
                    card.append(`Surprise: ${result.surprise}`);
                    card.setSuccess();

                    // Update activity feed
                    activityFeed.updateActivity(uploadActivityId, {
                        status: 'completed',
                        details: `Analysis complete. Surprise score: ${result.surprise}`,
                        data: {
                            files: [file.name],
                            analysis: result.analysis,
                            surprise: result.surprise
                        }
                    });

                    // Add surprise detection activity if high novelty
                    if (result.surprise > 0.5) {
                        console.log('⚡ [ROUTER] HIGH NOVELTY DETECTED! Surprise:', result.surprise);
                        activityFeed.addActivity({
                            id: `surprise_${Date.now()}`,
                            type: 'surprise_detected',
                            status: 'completed',
                            title: 'High novelty detected!',
                            details: `Surprise score: ${result.surprise.toFixed(2)}`,
                            timestamp: new Date().toISOString(),
                            data: { surprise: result.surprise }
                        });
                    }

                    // Assistant Response regarding file
                    const metadata = { surprise: result.surprise, research: result.surprise > 0.5 };
                    chat.addMessage('assistant', `Analyzed ${file.name}: ${result.analysis}`, metadata);

                } catch (err) {
                    console.error('❌ [ROUTER] File upload failed:', file.name, 'Error:', err);
                    card.append(`Error: ${err.message}`);
                    card.setError();
                    
                    activityFeed.updateActivity(uploadActivityId, {
                        status: 'error',
                        details: err.message
                    });
                    
                    chat.addMessage('assistant', `Failed to process ${file.name}.`);
                }
            }
        }

        // 2. Process Text
        if (text) {
            console.log('💬 [ROUTER] Processing text message:', text.substring(0, 100));
            chat.addMessage('user', text);
            messageHistory.push({ role: 'user', content: text });
            console.log('📚 [ROUTER] Message history length:', messageHistory.length);

            // Activity Feed: Log thinking process
            console.log('🧠 [ROUTER] Creating thinking activity...');
            const thinkingActivityId = activityFeed.addActivity({
                id: `thinking_${Date.now()}`,
                type: 'thinking',
                status: 'in_progress',
                title: 'Processing your request',
                details: 'Running active inference loop...',
                timestamp: new Date().toISOString(),
                data: {
                    reasoning: [
                        '1. Sensing: Parsing user intent',
                        '2. Retrieving: Searching mind for relevant knowledge',
                        '3. Reasoning: Applying logic and domain expertise',
                        '4. Embellishing: Adding emotional intelligence',
                        '5. Acting: Generating response',
                        '6. Learning: Storing to episodic memory'
                    ]
                }
            });

            // Brain Visualizer: Start thinking animation
            console.log('🎬 [ROUTER] Starting brain visualizer animation...');
            brainViz.startThinking();

            const card = visualizer.createCard("Thinking Process");
            card.append("User input received.");
            card.append("Initiating active inference cycle...");

            try {
                console.log('📡 [ROUTER] Sending message to API...');
                const response = await api.sendMessage(messageHistory);
                console.log('✅ [ROUTER] API response received:', response);
                const assistantMsg = response.choices[0].message;
                const metadata = response.metadata || {};
                console.log('📩 [ROUTER] Assistant message:', assistantMsg.content.substring(0, 100));
                console.log('📊 [ROUTER] Metadata:', metadata);

                card.append("Response generated.");
                card.setSuccess();

                // Update thinking activity
                console.log('📝 [ROUTER] Updating thinking activity to completed...');
                activityFeed.updateActivity(thinkingActivityId, {
                    status: 'completed',
                    details: 'Response generated successfully',
                    completed_at: new Date().toISOString(),
                    data: {
                        reasoning: metadata.reasoning_steps || [],
                        confidence: metadata.agent_state?.confidence || 0.85,
                        surprise: metadata.agent_state?.surprise_score || 0
                    }
                });

                // Process and display activity events from backend (tool creation, file changes, etc.)
                if (metadata.activity_events && metadata.activity_events.length > 0) {
                    console.log('🎯 [ROUTER] Processing activity events from backend:', metadata.activity_events);
                    metadata.activity_events.forEach(event => {
                        // Add each activity event to the feed
                        activityFeed.addActivity({
                            ...event,
                            status: 'completed',  // Mark as completed since it already happened
                            completed_at: new Date().toISOString()
                        });
                    });
                }

                // Brain visualizer: Update metrics
                const metrics = {
                    surprise_score: metadata.agent_state?.surprise_score || 0,
                    confidence: metadata.agent_state?.confidence || 0.85,
                    response_time: metadata.timing?.total_ms || 1500
                };
                console.log('📈 [ROUTER] Updating brain visualizer metrics:', metrics);
                brainViz.updateMetrics(metrics);

                // Stop thinking animation after slight delay
                setTimeout(() => {
                    console.log('⏸️ [ROUTER] Stopping brain visualizer animation...');
                    brainViz.stopThinking();
                }, 1500);

                // Activity Feed: Log memory update
                console.log('💾 [ROUTER] Logging memory update activity...');
                activityFeed.addActivity({
                    id: `memory_${Date.now()}`,
                    type: 'memory_update',
                    status: 'completed',
                    title: 'Interaction saved to episodic memory',
                    details: 'Learning from this conversation',
                    timestamp: new Date().toISOString(),
                    data: {}
                });

                chat.addMessage('assistant', assistantMsg.content, metadata);
                messageHistory.push(assistantMsg);

            } catch (err) {
                console.error('❌ [ROUTER] Message processing failed:', err);
                console.error('❌ [ROUTER] Error stack:', err.stack);
                card.append(`Error: ${err.message}`);
                card.setError();
                
                activityFeed.updateActivity(thinkingActivityId, {
                    status: 'error',
                    details: err.message
                });
                
                brainViz.stopThinking();
                
                chat.addMessage('assistant', "I encountered an error processing your request.");
            }
        }
    };

    // --- Event Listeners ---
    console.log('🎧 [ROUTER] Attaching event listeners...');

    chatForm.addEventListener('submit', handleSend);
    console.log('✅ [ROUTER] Chat form submit listener attached');

    fileBtn.addEventListener('click', () => {
        console.log('📎 [ROUTER] File button clicked');
        fileUploader.trigger();
    });
    console.log('✅ [ROUTER] File button listener attached');

    cameraBtn.addEventListener('click', () => {
        console.log('📷 [ROUTER] Camera button clicked (not implemented)');
        alert("Camera feature integration pending backend update.");
    });
    console.log('✅ [ROUTER] Camera button listener attached');

    micBtn.addEventListener('click', () => {
        console.log('🎤 [ROUTER] Microphone button clicked (not implemented)');
        alert("Microphone feature integration pending backend update.");
    });
    console.log('✅ [ROUTER] Microphone button listener attached');

    // Activity Feed Toggle
    if (activityFeedToggle) {
        activityFeedToggle.addEventListener('click', () => {
            activityFeedOpen = !activityFeedOpen;
            console.log('📊 [ROUTER] Activity feed toggle clicked - New state:', activityFeedOpen ? 'OPEN' : 'CLOSED');
            
            const feedWrapper = document.querySelector('.activity-feed-wrapper');
            
            if (activityFeedOpen) {
                console.log('✅ [ROUTER] Opening activity feed...');
                feedWrapper.classList.add('open');
                if (chatWrapper) chatWrapper.classList.add('feed-open');
                activityFeedToggle.classList.add('active');
            } else {
                console.log('⏸️ [ROUTER] Closing activity feed...');
                feedWrapper.classList.remove('open');
                if (chatWrapper) chatWrapper.classList.remove('feed-open');
                activityFeedToggle.classList.remove('active');
            }
        });
        console.log('✅ [ROUTER] Activity feed toggle listener attached');
    } else {
        console.warn('⚠️ [ROUTER] Activity feed toggle button not found!');
    }

    // Brain Visualizer Toggle
    if (brainVisualizerToggle) {
        brainVisualizerToggle.addEventListener('click', () => {
            brainVisualizerOpen = !brainVisualizerOpen;
            console.log('🧠 [ROUTER] Brain visualizer toggle clicked - New state:', brainVisualizerOpen ? 'OPEN' : 'CLOSED');
            
            const brainVizElement = document.querySelector('.brain-visualizer');
            
            if (brainVizElement) {
                if (brainVisualizerOpen) {
                    console.log('✅ [ROUTER] Opening brain visualizer...');
                    brainVizElement.classList.add('open');
                    brainVisualizerToggle.classList.add('active');
                } else {
                    console.log('⏸️ [ROUTER] Closing brain visualizer...');
                    brainVizElement.classList.remove('open');
                    brainVisualizerToggle.classList.remove('active');
                }
            } else {
                console.warn('⚠️ [ROUTER] Brain visualizer element not found!');
            }
        });
        console.log('✅ [ROUTER] Brain visualizer toggle listener attached');
    } else {
        console.warn('⚠️ [ROUTER] Brain visualizer toggle button not found!');
    }
    
    console.log('🎉 [ROUTER] All initialization complete! AetherMind is ready.');
});

function updateDomainIndicator(domain) {
    console.log('🎯 [ROUTER] updateDomainIndicator called with domain:', domain);
    
    const domainText = document.getElementById('domain-text');
    const domainIndicator = document.getElementById('domain-indicator');
    
    if (!domainText || !domainIndicator) {
        console.warn('⚠️ [ROUTER] Domain indicator elements not found!');
        return;
    }
    
    const domainConfig = {
        'code': { name: 'Software Dev', icon: 'fas fa-code', color: '#8b5cf6' },
        'research': { name: 'Research', icon: 'fas fa-microscope', color: '#f59e0b' },
        'business': { name: 'Business', icon: 'fas fa-briefcase', color: '#3b82f6' },
        'legal': { name: 'Legal', icon: 'fas fa-balance-scale', color: '#ec4899' },
        'finance': { name: 'Finance', icon: 'fas fa-chart-line', color: '#10b981' },
        'general': { name: 'Multi-Domain', icon: 'fas fa-star', color: '#10b981' }
    };
    
    const config = domainConfig[domain] || domainConfig['general'];
    console.log('⚙️ [ROUTER] Domain config selected:', config);
    
    domainText.textContent = config.name;
    const icon = domainIndicator.querySelector('i');
    if (icon) {
        icon.className = config.icon;
        icon.style.color = config.color;
        console.log('✅ [ROUTER] Domain icon updated:', config.icon);
    } else {
        console.warn('⚠️ [ROUTER] Domain icon element not found!');
    }
    domainIndicator.style.borderColor = config.color + '50';
    domainIndicator.style.background = config.color + '20';
    domainText.style.color = config.color;
    console.log('✅ [ROUTER] Domain indicator styling applied');
}
