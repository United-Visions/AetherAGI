// shell-router.js - Minimal Shell Entry Point
// Agent-controlled UI with background task architecture

import { api, setApiKeyModal } from './api.js';
import { UIOrchestrator } from './core/UIOrchestrator.js';
import { OnboardingAgent } from './core/OnboardingAgent.js';
import { NotificationManager } from './core/NotificationManager.js';
import { BackgroundTaskManager } from './core/BackgroundTaskManager.js';
import { VoiceManager } from './core/VoiceManager.js';
import { ApiKeyModal } from './components/ApiKeyModal.js';

class AetherShell {
    constructor() {
        this.config = window.AETHER_CONFIG || {};
        this.messageHistory = [];
        this.isOnboarded = false;
        
        // Core managers
        this.ui = null;
        this.notifications = null;
        this.tasks = null;
        this.onboarding = null;
        this.voice = null;
    }

    async init() {
        console.log('🚀 [SHELL] Initializing AetherMind Shell...');
        
        // Initialize API Key Modal
        const apiKeyModal = new ApiKeyModal();
        setApiKeyModal(apiKeyModal);
        window.apiKeyModal = apiKeyModal;
        
        // Check for API key
        const apiKey = localStorage.getItem('aethermind_api_key');
        if (!apiKey) {
            console.warn('⚠️ [SHELL] No API key found, showing modal...');
            apiKeyModal.show();
        }
        
        // Initialize core managers
        this.notifications = new NotificationManager();
        this.tasks = new BackgroundTaskManager();
        this.ui = new UIOrchestrator(this);
        this.onboarding = new OnboardingAgent(this);
        this.voice = new VoiceManager(this);
        
        // Set up DOM references
        this.setupDOM();
        
        // Check user profile
        await this.checkUserProfile();
        
        // Set up admin panel for pilot users
        if (this.config.isPilotUser) {
            this.setupAdminPanel();
        }
        
        // Start the experience
        if (!this.isOnboarded) {
            this.onboarding.start();
        } else {
            this.showWelcomeBack();
        }
        
        // Make globally accessible
        window.aether = this;
        
        console.log('✅ [SHELL] Shell initialized');
    }

    setupDOM() {
        // Core elements
        this.elements = {
            messagesContainer: document.getElementById('messages'),
            chatForm: document.getElementById('chat-form'),
            chatInput: document.getElementById('chat-input'),
            sendBtn: document.getElementById('send-btn'),
            fileInput: document.getElementById('file-input'),
            filePreviews: document.getElementById('file-previews'),
            quickActions: document.getElementById('quick-actions'),
            notificationBtn: document.getElementById('notification-btn'),
            notificationBadge: document.getElementById('notification-badge'),
            notificationPanel: document.getElementById('notification-panel'),
            adminBtn: document.getElementById('admin-panel-btn'),
            adminPanel: document.getElementById('admin-panel'),
            tasksIndicator: document.getElementById('tasks-indicator'),
            tasksCount: document.getElementById('tasks-count'),
            dynamicSlots: document.getElementById('dynamic-slots'),
            toastContainer: document.getElementById('toast-container')
        };
        
        // Event listeners
        this.elements.chatForm.addEventListener('submit', (e) => this.handleSend(e));
        this.elements.chatInput.addEventListener('input', () => this.handleInputChange());
        this.elements.chatInput.addEventListener('keydown', (e) => this.handleKeydown(e));
        
        // Notification panel toggle
        this.elements.notificationBtn.addEventListener('click', () => this.toggleNotifications());
        document.getElementById('close-notifications')?.addEventListener('click', () => this.toggleNotifications(false));
        
        // Admin panel toggle (pilot users only)
        this.elements.adminBtn?.addEventListener('click', () => this.toggleAdminPanel());
        document.getElementById('close-admin')?.addEventListener('click', () => this.toggleAdminPanel(false));
        
        // Tasks indicator click
        this.elements.tasksIndicator?.addEventListener('click', () => this.showTasksPanel());
        
        // File handling
        document.getElementById('file-btn')?.addEventListener('click', () => this.elements.fileInput.click());
        this.elements.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Camera and voice (lazy load when needed)
        document.getElementById('camera-btn')?.addEventListener('click', () => this.ui.activateCamera());
        document.getElementById('voice-btn')?.addEventListener('click', () => this.toggleVoice());
    }

    toggleVoice() {
        const enabled = this.voice.toggle();
        this.toast(enabled ? '🔊 Voice enabled' : '🔇 Voice muted', 'info');
    }

    async checkUserProfile() {
        try {
            const profile = await api.getUserProfile();
            if (profile && profile.onboarded) {
                this.isOnboarded = true;
                this.userProfile = profile;
            }
        } catch (err) {
            console.log('📝 [SHELL] No existing profile, starting fresh');
            this.isOnboarded = false;
        }
    }

    async saveProfile() {
        try {
            await api.saveUserProfile(this.userProfile);
            localStorage.setItem('aethermind_profile', JSON.stringify(this.userProfile));
            console.log('💾 [SHELL] Profile saved');
        } catch (err) {
            console.warn('⚠️ [SHELL] Could not save profile:', err);
            // Save locally as fallback
            localStorage.setItem('aethermind_profile', JSON.stringify(this.userProfile));
        }
    }

    handleInputChange() {
        const hasContent = this.elements.chatInput.value.trim().length > 0;
        this.elements.sendBtn.disabled = !hasContent;
        
        // Auto-resize textarea
        this.elements.chatInput.style.height = 'auto';
        this.elements.chatInput.style.height = Math.min(this.elements.chatInput.scrollHeight, 150) + 'px';
    }

    handleKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!this.elements.sendBtn.disabled) {
                this.handleSend(e);
            }
        }
    }

    async handleSend(e) {
        e.preventDefault();
        
        const text = this.elements.chatInput.value.trim();
        if (!text) return;
        
        // Clear input
        this.elements.chatInput.value = '';
        this.elements.sendBtn.disabled = true;
        this.elements.chatInput.style.height = 'auto';
        
        // Add user message
        this.addMessage('user', text);
        this.messageHistory.push({ role: 'user', content: text });
        
        // Check if still onboarding
        if (!this.isOnboarded) {
            await this.onboarding.handleResponse(text);
            return;
        }
        
        // Show typing indicator
        const typingId = this.showTyping();
        
        try {
            // Send to backend with profile context (includes personas)
            const response = await api.chat({
                messages: this.messageHistory,
                context: {
                    currentProfile: this.userProfile
                }
            });
            
            // Remove typing indicator
            this.removeTyping(typingId);
            
            // Process response
            const assistantMsg = response.choices[0].message;
            const metadata = response.metadata || {};
            
            // Check for UI commands from agent
            if (metadata.ui_commands) {
                await this.ui.executeCommands(metadata.ui_commands);
            }
            
            // Check for background tasks
            if (metadata.background_tasks) {
                metadata.background_tasks.forEach(task => {
                    this.tasks.addTask(task);
                });
            }
            
            // Check for persona changes
            if (metadata.save_persona) {
                if (!this.userProfile.personas) this.userProfile.personas = {};
                this.userProfile.personas[metadata.save_persona.name] = metadata.save_persona;
                await this.saveProfile();
                this.toast(`Persona "${metadata.save_persona.name}" saved! 🎭`, 'success');
            }
            
            if (metadata.hasOwnProperty('switch_persona')) {
                this.userProfile.activePersona = metadata.switch_persona;
                await this.saveProfile();
                if (metadata.switch_persona) {
                    this.toast(`Switched to ${metadata.switch_persona} persona 🎭`, 'info');
                } else {
                    this.toast('Back to normal mode 🎭', 'info');
                }
            }
            
            // Display response
            this.addMessage('assistant', assistantMsg.content, metadata);
            this.messageHistory.push(assistantMsg);
            
            // Speak the response if voice is enabled
            if (this.voice?.enabled) {
                this.voice.speak(assistantMsg.content);
            }
            
        } catch (err) {
            this.removeTyping(typingId);
            this.addMessage('assistant', "I'm having trouble connecting. Let me try again in a moment.");
            console.error('❌ [SHELL] Send error:', err);
        }
    }

    addMessage(role, content, metadata = {}) {
        const row = document.createElement('div');
        row.className = `message-row ${role}`;
        
        if (role === 'user') {
            row.innerHTML = `<div class="user-bubble">${this.escapeHtml(content)}</div>`;
        } else {
            const formattedContent = this.formatMessage(content);
            row.innerHTML = `
                <div class="assistant-content">
                    <div class="assistant-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="assistant-text">${formattedContent}</div>
                </div>
            `;
        }
        
        this.elements.messagesContainer.appendChild(row);
        this.scrollToBottom();
        
        return row;
    }

    addInteractiveMessage(role, content, actions = []) {
        const row = this.addMessage(role, content);
        
        if (actions.length > 0) {
            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'message-actions';
            
            actions.forEach(action => {
                const chip = document.createElement('button');
                chip.className = `action-chip ${action.primary ? 'primary' : 'secondary'}`;
                chip.textContent = action.label;
                chip.onclick = () => action.handler();
                actionsDiv.appendChild(chip);
            });
            
            row.querySelector('.assistant-text, .user-bubble')?.appendChild(actionsDiv);
        }
        
        return row;
    }

    showTyping() {
        const id = `typing_${Date.now()}`;
        const row = document.createElement('div');
        row.className = 'message-row assistant';
        row.id = id;
        row.innerHTML = `
            <div class="assistant-content">
                <div class="assistant-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="assistant-text">
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
        `;
        this.elements.messagesContainer.appendChild(row);
        this.scrollToBottom();
        return id;
    }

    removeTyping(id) {
        document.getElementById(id)?.remove();
    }

    async showWelcomeBack() {
        // Use the real agent to welcome the user back
        const name = this.userProfile?.learnedFacts?.name || this.userProfile?.name || 'there';
        
        const typingId = this.showTyping();
        
        try {
            const response = await api.chat({
                messages: [],
                context: {
                    mode: 'welcome_back',
                    isOnboarding: false,
                    currentProfile: this.userProfile
                }
            });
            
            this.removeTyping(typingId);
            
            let assistantMsg = response.choices?.[0]?.message?.content || 
                `${this.getTimeBasedGreeting()}, ${name}! 👋 Good to see you again. What would you like to work on?`;
            
            this.addMessage('assistant', assistantMsg);
            this.messageHistory.push({ role: 'assistant', content: assistantMsg });
            
            // Speak welcome if voice enabled
            if (this.voice?.enabled) {
                this.voice.speak(assistantMsg);
            }
            
        } catch (err) {
            this.removeTyping(typingId);
            // Fallback to simple greeting if API fails
            const greeting = this.getTimeBasedGreeting();
            const fallbackMsg = `${greeting}, ${name}! 👋 Good to see you again. What would you like to work on today?`;
            this.addMessage('assistant', fallbackMsg);
            
            if (this.voice?.enabled) {
                this.voice.speak(fallbackMsg);
            }
        }
    }

    getTimeBasedGreeting() {
        const hour = new Date().getHours();
        if (hour < 12) return 'Good morning';
        if (hour < 17) return 'Good afternoon';
        return 'Good evening';
    }

    formatMessage(content) {
        // Extract and handle <think> tags first - animate as thinking stream
        let thinkingContent = '';
        let mainContent = content;
        
        const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);
        if (thinkMatch) {
            thinkingContent = thinkMatch[1].trim();
            mainContent = content.replace(/<think>[\s\S]*?<\/think>/, '').trim();
        }
        
        // Basic markdown-like formatting on main content
        let formatted = this.escapeHtml(mainContent);
        
        // Bold
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Code blocks
        formatted = formatted.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        
        // Inline code
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');
        
        // If there was thinking content, add collapsible thinking section
        if (thinkingContent) {
            const thinkingHtml = `
                <details class="thinking-stream">
                    <summary class="thinking-toggle">
                        <span class="thinking-icon">💭</span> View thinking process
                    </summary>
                    <div class="thinking-content">${this.escapeHtml(thinkingContent).replace(/\n/g, '<br>')}</div>
                </details>
            `;
            formatted = thinkingHtml + formatted;
        }
        
        return formatted;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
    }

    toggleNotifications(show = null) {
        const panel = this.elements.notificationPanel;
        const shouldShow = show !== null ? show : !panel.classList.contains('open');
        
        if (shouldShow) {
            panel.classList.remove('hidden');
            setTimeout(() => panel.classList.add('open'), 10);
            this.notifications.markAllRead();
        } else {
            panel.classList.remove('open');
            setTimeout(() => panel.classList.add('hidden'), 250);
        }
    }

    toggleAdminPanel(show = null) {
        const panel = this.elements.adminPanel;
        const shouldShow = show !== null ? show : !panel.classList.contains('open');
        
        if (shouldShow) {
            panel.classList.remove('hidden');
            setTimeout(() => panel.classList.add('open'), 10);
        } else {
            panel.classList.remove('open');
            setTimeout(() => panel.classList.add('hidden'), 250);
        }
    }

    setupAdminPanel() {
        this.elements.adminBtn.classList.remove('hidden');
        
        // Load benchmark controls
        this.loadBenchmarkControls();
    }

    async loadBenchmarkControls() {
        const container = document.getElementById('benchmark-controls');
        if (!container) return;
        
        container.innerHTML = `
            <button class="action-chip primary" onclick="window.aether.runBenchmark('gsm')">
                <i class="fas fa-calculator"></i> Run GSM
            </button>
            <button class="action-chip primary" onclick="window.aether.runBenchmark('mmlu')">
                <i class="fas fa-book"></i> Run MMLU
            </button>
            <button class="action-chip" onclick="window.aether.runBenchmark('humaneval')">
                <i class="fas fa-code"></i> Run HumanEval
            </button>
        `;
    }

    async runBenchmark(type) {
        this.toast(`Starting ${type.toUpperCase()} benchmark...`, 'info');
        
        // Add as background task
        this.tasks.addTask({
            id: `benchmark_${type}_${Date.now()}`,
            type: 'benchmark',
            name: `${type.toUpperCase()} Benchmark`,
            status: 'running'
        });
        
        try {
            await api.runBenchmark(type);
        } catch (err) {
            this.toast(`Benchmark failed: ${err.message}`, 'error');
        }
    }

    showTasksPanel() {
        // Show tasks in notification panel for now
        this.notifications.showTasks(this.tasks.getTasks());
        this.toggleNotifications(true);
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;
        
        // Show quick actions if hidden
        this.elements.quickActions.classList.remove('hidden');
        this.elements.filePreviews.classList.remove('hidden');
        
        // Add previews
        files.forEach(file => {
            this.addFilePreview(file);
        });
    }

    addFilePreview(file) {
        const preview = document.createElement('div');
        preview.className = 'file-preview';
        
        let icon = 'fa-file';
        if (file.type.startsWith('image/')) icon = 'fa-image';
        if (file.type.startsWith('video/')) icon = 'fa-video';
        if (file.type.startsWith('audio/')) icon = 'fa-music';
        
        preview.innerHTML = `
            <i class="fas ${icon}"></i>
            <span>${file.name}</span>
            <i class="fas fa-times remove-file"></i>
        `;
        
        preview.querySelector('.remove-file').onclick = () => {
            preview.remove();
            if (this.elements.filePreviews.children.length === 0) {
                this.elements.filePreviews.classList.add('hidden');
            }
        };
        
        this.elements.filePreviews.appendChild(preview);
    }

    toast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <i class="fas ${type === 'success' ? 'fa-check' : type === 'error' ? 'fa-exclamation' : 'fa-info'}"></i>
            <span>${message}</span>
        `;
        
        this.elements.toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 4000);
    }

    // API for agent to call
    async agentAction(action, params = {}) {
        switch (action) {
            case 'show_quick_actions':
                this.elements.quickActions.classList.remove('hidden');
                break;
            case 'hide_quick_actions':
                this.elements.quickActions.classList.add('hidden');
                break;
            case 'show_tasks':
                this.showTasksPanel();
                break;
            case 'add_notification':
                this.notifications.add(params);
                break;
            case 'add_task':
                this.tasks.addTask(params);
                break;
            case 'update_ui':
                await this.ui.executeCommands(params.commands);
                break;
            case 'request_media':
                await this.ui.requestMedia(params.type, params.reason);
                break;
            default:
                console.warn('Unknown agent action:', action);
        }
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    const shell = new AetherShell();
    shell.init();
});
