// VoiceManager.js - Handles voice synthesis and playback
// Integrates with Edge TTS backend for AI voice output

import { api } from '../api.js';

export class VoiceManager {
    constructor(shell) {
        this.shell = shell;
        this.enabled = false;
        this.isPlaying = false;
        this.audioQueue = [];
        this.currentAudio = null;
        
        // Settings
        this.settings = {
            autoPlay: true,          // Auto-play responses
            persona: null,           // Use persona-specific voice
            voiceId: null,           // Override voice ID
            rate: '+0%',
            pitch: '+0Hz'
        };
        
        // Load saved settings
        this.loadSettings();
        
        console.log('🔊 [Voice] VoiceManager initialized');
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('aethermind_voice_settings');
            if (saved) {
                this.settings = { ...this.settings, ...JSON.parse(saved) };
                this.enabled = this.settings.autoPlay;
            }
        } catch (e) {
            console.warn('⚠️ [Voice] Could not load settings:', e);
        }
    }

    saveSettings() {
        try {
            localStorage.setItem('aethermind_voice_settings', JSON.stringify(this.settings));
        } catch (e) {
            console.warn('⚠️ [Voice] Could not save settings:', e);
        }
    }

    toggle() {
        this.enabled = !this.enabled;
        this.settings.autoPlay = this.enabled;
        this.saveSettings();
        
        // Update UI
        this.updateVoiceButton();
        
        // Stop current audio if disabling
        if (!this.enabled) {
            this.stop();
        }
        
        console.log(`🔊 [Voice] Voice ${this.enabled ? 'enabled' : 'disabled'}`);
        return this.enabled;
    }

    updateVoiceButton() {
        const btn = document.getElementById('voice-btn');
        if (btn) {
            btn.classList.toggle('active', this.enabled);
            btn.title = this.enabled ? 'Voice On (click to mute)' : 'Voice Off (click to enable)';
            
            // Update icon
            const icon = btn.querySelector('i');
            if (icon) {
                icon.className = this.enabled ? 'fas fa-volume-up' : 'fas fa-volume-mute';
            }
        }
    }

    async speak(text, options = {}) {
        if (!this.enabled && !options.force) {
            return;
        }
        
        // Skip empty text
        if (!text || !text.trim()) {
            return;
        }
        
        console.log('🔊 [Voice] Speaking:', text.substring(0, 50) + '...');
        
        try {
            // Get current persona if active
            const persona = options.persona || 
                           this.shell.userProfile?.activePersona || 
                           this.settings.persona;
            
            const response = await api.synthesizeVoice(text, {
                persona: persona,
                voiceId: options.voiceId || this.settings.voiceId,
                rate: options.rate || this.settings.rate,
                pitch: options.pitch || this.settings.pitch
            });
            
            if (response.success && response.audio) {
                await this.playAudio(response.audio);
            }
        } catch (error) {
            console.error('❌ [Voice] Speech synthesis error:', error);
            // Silently fail - don't break the UI
        }
    }

    async playAudio(base64Audio) {
        return new Promise((resolve, reject) => {
            try {
                // Create audio from base64
                const audioBlob = this.base64ToBlob(base64Audio, 'audio/mp3');
                const audioUrl = URL.createObjectURL(audioBlob);
                
                // Create audio element
                const audio = new Audio(audioUrl);
                this.currentAudio = audio;
                this.isPlaying = true;
                
                // Update UI to show playing state
                this.updatePlayingState(true);
                
                audio.onended = () => {
                    URL.revokeObjectURL(audioUrl);
                    this.isPlaying = false;
                    this.currentAudio = null;
                    this.updatePlayingState(false);
                    this.playNextInQueue();
                    resolve();
                };
                
                audio.onerror = (e) => {
                    URL.revokeObjectURL(audioUrl);
                    this.isPlaying = false;
                    this.currentAudio = null;
                    this.updatePlayingState(false);
                    console.error('❌ [Voice] Audio playback error:', e);
                    reject(e);
                };
                
                audio.play().catch(err => {
                    console.warn('⚠️ [Voice] Autoplay blocked:', err);
                    // Some browsers block autoplay
                    this.isPlaying = false;
                    this.currentAudio = null;
                    this.updatePlayingState(false);
                    resolve();
                });
                
            } catch (error) {
                this.isPlaying = false;
                reject(error);
            }
        });
    }

    base64ToBlob(base64, mimeType) {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }

    updatePlayingState(playing) {
        const btn = document.getElementById('voice-btn');
        if (btn) {
            btn.classList.toggle('playing', playing);
            
            const icon = btn.querySelector('i');
            if (icon && this.enabled) {
                icon.className = playing ? 'fas fa-volume-up fa-beat' : 'fas fa-volume-up';
            }
        }
    }

    queueSpeech(text, options = {}) {
        if (this.isPlaying) {
            this.audioQueue.push({ text, options });
        } else {
            this.speak(text, options);
        }
    }

    playNextInQueue() {
        if (this.audioQueue.length > 0) {
            const next = this.audioQueue.shift();
            this.speak(next.text, next.options);
        }
    }

    stop() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio = null;
        }
        this.isPlaying = false;
        this.audioQueue = [];
        this.updatePlayingState(false);
    }

    setVoice(voiceId) {
        this.settings.voiceId = voiceId;
        this.saveSettings();
        console.log('🔊 [Voice] Voice set to:', voiceId);
    }

    setRate(rate) {
        this.settings.rate = rate;
        this.saveSettings();
    }

    setPitch(pitch) {
        this.settings.pitch = pitch;
        this.saveSettings();
    }

    // Get available voice profiles for UI
    async getVoiceProfiles() {
        try {
            const data = await api.listVoices();
            return {
                voices: data.voices || [],
                profiles: data.profiles || {}
            };
        } catch (error) {
            console.error('❌ [Voice] Could not get voices:', error);
            return { voices: [], profiles: {} };
        }
    }
}
