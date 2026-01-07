// OnboardingAgent.js - Real AI Conversation for Onboarding
// Agent actually talks to backend, learns organically, saves everything step by step

import { api } from '../api.js';

export class OnboardingAgent {
    constructor(shell) {
        this.shell = shell;
        
        // Conversation history for context
        this.conversationHistory = [];
        
        // Profile data accumulated through conversation
        this.profile = {
            onboarded: false,
            conversationLog: [],
            learnedFacts: {},
            preferences: {},
            mediaProvided: false,
            askLater: [],
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            // Persona vector - different personalities user can switch between
            personas: {},
            activePersona: null
        };
    }

    async start() {
        console.log('🎭 [ONBOARDING] Starting real conversation with agent...');
        
        // Check if we should resume a previous onboarding
        const resumed = await this.resumeIfNeeded();
        if (resumed) return;
        
        // Send the initial message to get the agent to introduce herself
        await this.sendToAgent(null, 'onboarding_start');
    }

    async handleResponse(text) {
        // User responded - send to the real agent
        await this.sendToAgent(text, 'onboarding');
    }

    async sendToAgent(userMessage, mode = 'onboarding') {
        // Build message history with onboarding context
        const messages = [...this.conversationHistory];
        
        if (userMessage) {
            messages.push({ role: 'user', content: userMessage });
            this.conversationHistory.push({ role: 'user', content: userMessage });
            
            // Save user message to profile log
            this.profile.conversationLog.push({
                role: 'user',
                content: userMessage,
                timestamp: new Date().toISOString()
            });
            
            // Also save incrementally to localStorage as backup
            this.saveToLocalStorage();
        }
        
        // Show typing indicator
        const typingId = this.shell.showTyping();
        
        try {
            // Call the real backend with onboarding context
            const response = await api.chat({
                messages: messages,
                context: {
                    mode: mode,
                    isOnboarding: true,
                    currentProfile: this.profile,
                    timezone: this.profile.timezone
                }
            });
            
            this.shell.removeTyping(typingId);
            
            // Extract the response content
            let assistantMsg = '';
            let metadata = {};
            
            if (response.choices && response.choices[0]) {
                assistantMsg = response.choices[0].message?.content || '';
                metadata = response.metadata || {};
            } else if (response.content) {
                assistantMsg = response.content;
                metadata = response.metadata || {};
            } else if (typeof response === 'string') {
                assistantMsg = response;
            }
            
            // Clean any control markers from the response
            assistantMsg = assistantMsg
                .replace('[ONBOARDING_COMPLETE]', '')
                .replace('[REQUEST_MEDIA]', '')
                .trim();
            
            // Add to conversation history
            this.conversationHistory.push({ role: 'assistant', content: assistantMsg });
            
            // Save assistant message to profile log
            this.profile.conversationLog.push({
                role: 'assistant',
                content: assistantMsg,
                timestamp: new Date().toISOString()
            });
            
            // Check if agent extracted any facts about the user
            if (metadata.learned_facts) {
                Object.assign(this.profile.learnedFacts, metadata.learned_facts);
                console.log('📚 [ONBOARDING] Learned facts:', metadata.learned_facts);
            }
            
            // Check for user preferences the agent detected
            if (metadata.user_preferences) {
                Object.assign(this.profile.preferences, metadata.user_preferences);
            }
            
            // Check for persona updates
            if (metadata.save_persona) {
                this.profile.personas[metadata.save_persona.name] = metadata.save_persona;
                console.log('🎭 [PERSONA] Saved:', metadata.save_persona.name);
            }
            
            if (metadata.switch_persona) {
                this.profile.activePersona = metadata.switch_persona;
                console.log('🎭 [PERSONA] Switched to:', metadata.switch_persona);
            }
            
            // Check if agent wants to request media
            if (metadata.request_media) {
                this.showMediaRequest(assistantMsg, metadata.request_media);
                await this.saveProgress();
                return;
            }
            
            // Check if onboarding is complete (agent decides)
            if (metadata.onboarding_complete) {
                await this.completeOnboarding(assistantMsg);
                return;
            }
            
            // Check for interactive actions the agent wants to show
            if (metadata.actions && metadata.actions.length > 0) {
                const actions = metadata.actions.map(a => ({
                    label: a.label,
                    primary: a.primary || false,
                    handler: () => this.handleAction(a)
                }));
                this.shell.addInteractiveMessage('assistant', assistantMsg, actions);
            } else {
                // Regular message
                this.shell.addMessage('assistant', assistantMsg);
            }
            
            // Save progress after each exchange
            await this.saveProgress();
            
        } catch (err) {
            this.shell.removeTyping(typingId);
            console.error('❌ [ONBOARDING] Error talking to agent:', err);
            
            // Fallback - still try to be helpful and keep the conversation going
            this.shell.addMessage('assistant', 
                "I'm having a bit of trouble connecting right now, but don't worry! Tell me about yourself and what you're hoping to accomplish - I'm listening. 💫"
            );
        }
    }

    showMediaRequest(message, mediaType) {
        this.shell.addInteractiveMessage('assistant', message, [
            { 
                label: '📷 Share photo', 
                primary: true, 
                handler: () => this.provideMedia('photo') 
            },
            { 
                label: '🎥 Share video', 
                handler: () => this.provideMedia('video') 
            },
            { 
                label: 'Maybe later', 
                handler: () => this.declineMedia() 
            }
        ]);
    }

    async provideMedia(type) {
        this.profile.mediaProvided = true;
        this.profile.mediaType = type;
        
        // Activate camera through UI orchestrator
        if (this.shell.ui && this.shell.ui.activateCamera) {
            await this.shell.ui.activateCamera(type);
        }
        
        await this.saveProgress();
        
        // Continue conversation - let agent know user shared media
        await this.sendToAgent(`[I'm sharing a ${type} now]`, 'onboarding');
    }

    async declineMedia() {
        this.profile.askLater.push({
            type: 'media',
            declinedAt: new Date().toISOString()
        });
        
        // Display user's choice
        this.shell.addMessage('user', 'Maybe later');
        
        await this.saveProgress();
        
        // Let the agent respond naturally to the decline
        await this.sendToAgent("[User said they'd prefer to share media later, that's fine]", 'onboarding');
    }

    async handleAction(action) {
        // User clicked an action button the agent provided
        const userResponse = action.response || action.label;
        
        // Save any data from the action
        if (action.key && action.value) {
            this.profile.learnedFacts[action.key] = action.value;
        }
        
        // Display user's choice
        this.shell.addMessage('user', userResponse);
        
        // Continue conversation
        await this.sendToAgent(userResponse, 'onboarding');
    }

    saveToLocalStorage() {
        try {
            localStorage.setItem('aethermind_onboarding_profile', JSON.stringify(this.profile));
        } catch (e) {
            console.warn('Could not save to localStorage:', e);
        }
    }

    async saveProgress() {
        this.saveToLocalStorage();
        
        try {
            await api.saveUserProfile(this.profile);
            console.log('💾 [ONBOARDING] Progress saved to backend');
        } catch (err) {
            console.warn('⚠️ [ONBOARDING] Could not save to backend (will retry):', err.message);
        }
    }

    async completeOnboarding(finalMessage) {
        this.profile.onboarded = true;
        this.profile.onboardedAt = new Date().toISOString();
        
        // Save final profile
        this.saveToLocalStorage();
        
        try {
            await api.saveUserProfile(this.profile);
        } catch (err) {
            console.warn('Profile save failed, stored locally:', err);
        }
        
        // Update shell state
        this.shell.isOnboarded = true;
        this.shell.userProfile = this.profile;
        
        // Show the agent's final message
        this.shell.addMessage('assistant', finalMessage);
        
        // Show quick actions now that onboarding is complete
        if (this.shell.elements?.quickActions) {
            this.shell.elements.quickActions.classList.remove('hidden');
        }
        
        console.log('✅ [ONBOARDING] Complete! Profile:', this.profile.learnedFacts);
    }

    // Resume from saved state if user returns mid-onboarding
    async resumeIfNeeded() {
        try {
            const saved = localStorage.getItem('aethermind_onboarding_profile');
            if (saved) {
                const parsed = JSON.parse(saved);
                if (!parsed.onboarded && parsed.conversationLog?.length > 0) {
                    console.log('🔄 [ONBOARDING] Resuming previous conversation...');
                    this.profile = parsed;
                    this.conversationHistory = parsed.conversationLog.map(msg => ({
                        role: msg.role,
                        content: msg.content
                    }));
                    
                    // Let agent know we're resuming
                    await this.sendToAgent("[User has returned, we were in the middle of getting to know each other]", 'onboarding_resume');
                    return true;
                }
            }
        } catch (e) {
            console.warn('Could not resume onboarding:', e);
        }
        return false;
    }

    // Called later to re-ask skipped questions
    async askLaterQuestions() {
        const mediaAsk = this.profile.askLater.find(q => q.type === 'media');
        
        if (mediaAsk) {
            // Check if enough time has passed (e.g., 1 hour)
            const declinedAt = new Date(mediaAsk.declinedAt);
            const hoursPassed = (Date.now() - declinedAt.getTime()) / (1000 * 60 * 60);
            
            if (hoursPassed >= 1) {
                // Remove from queue
                this.profile.askLater = this.profile.askLater.filter(q => q.type !== 'media');
                
                // Ask again through natural conversation
                const name = this.profile.learnedFacts.name || 'there';
                
                this.shell.addInteractiveMessage('assistant', 
                    `Hey ${name}! I was thinking - a photo would really help me personalize things for you. No pressure at all though! 😊`,
                    [
                        { label: '📸 Sure, let\'s do it', primary: true, handler: () => this.provideMedia('photo') },
                        { label: 'Not right now', handler: () => {
                            this.shell.addMessage('assistant', 'No worries! I\'ll focus on getting to know you through our conversations instead. 💬');
                        }}
                    ]
                );
            }
        }
    }
}
