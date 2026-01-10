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
            persona: 'aethermind',   // Default Aether voice (professional, clear, friendly)
            voiceId: null,           // Override voice ID
            rate: '+0%',
            pitch: '+0Hz'
        };
        
        // Load saved settings
        this.loadSettings();
        
        console.log('🔊 [Voice] VoiceManager initialized');
    }

    isEnabled() {
        return this.enabled;
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
            console.log('🔇 [Voice] Voice disabled, skipping speech');
            return;
        }
        
        // Skip empty text
        if (!text || !text.trim()) {
            console.log('⚠️ [Voice] Empty text, skipping');
            return;
        }
        
        // Clean text before sending (remove think tags, excessive formatting)
        const cleanText = this.cleanTextForSpeech(text);
        
        if (!cleanText || !cleanText.trim()) {
            console.log('⚠️ [Voice] Text empty after cleaning, skipping');
            return;
        }
        
        console.log('🔊 [Voice] Speaking:', cleanText.substring(0, 100) + '...');
        console.log('🔊 [Voice] Enabled:', this.enabled, 'Settings:', this.settings);
        
        try {
            // Get current persona if active, fallback to Aether's default voice
            const persona = options.persona || 
                           this.shell.userProfile?.activePersona || 
                           this.settings.persona ||
                           'aethermind';  // Aether's signature voice
            
            console.log('🎤 [Voice] Calling API with persona:', persona);
            
            const response = await api.synthesizeVoice(cleanText, {
                persona: persona,
                voiceId: options.voiceId || this.settings.voiceId,
                rate: options.rate || this.settings.rate,
                pitch: options.pitch || this.settings.pitch
            });
            
            console.log('📡 [Voice] API response:', response);
            
            if (response && response.success && response.audio) {
                console.log('🎵 [Voice] Playing audio, size:', response.audio.length, 'bytes');
                await this.playAudio(response.audio);
                console.log('✅ [Voice] Playback complete');
            } else {
                console.warn('⚠️ [Voice] No audio in response:', response);
            }
        } catch (error) {
            console.error('❌ [Voice] Speech synthesis error:', error);
            console.error('❌ [Voice] Error stack:', error.stack);
            // Show user-friendly error
            if (this.shell && this.shell.toast) {
                this.shell.toast('⚠️ Voice playback failed', 'warning');
            }
        }
    }

    cleanTextForSpeech(text) {
        // Remove <think> tags and content
        text = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
        
        // Remove code blocks
        text = text.replace(/```[\s\S]*?```/g, '');
        text = text.replace(/`[^`]+`/g, '');
        
        // Remove markdown links but keep text
        text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
        
        // Remove markdown formatting
        text = text.replace(/[*_]{1,2}([^*_]+)[*_]{1,2}/g, '$1');
        
        // Remove headers
        text = text.replace(/^#{1,6}\s+/gm, '');
        
        // Clean up whitespace
        text = text.replace(/\n{3,}/g, '\n\n');
        text = text.replace(/\s+/g, ' ');
        
        return text.trim();
    }

    async playAudio(base64Audio) {
        return new Promise((resolve, reject) => {
            try {
                console.log('🎵 [Voice] Converting base64 to audio blob...');
                
                // Create audio from base64
                const audioBlob = this.base64ToBlob(base64Audio, 'audio/mp3');
                const audioUrl = URL.createObjectURL(audioBlob);
                
                console.log('🎵 [Voice] Created audio URL:', audioUrl, 'Blob size:', audioBlob.size);
                
                // Create audio element
                const audio = new Audio(audioUrl);
                this.currentAudio = audio;
                this.isPlaying = true;
                
                // Update UI to show playing state
                this.updatePlayingState(true);
                
                audio.onended = () => {
                    console.log('✅ [Voice] Audio playback ended');
                    URL.revokeObjectURL(audioUrl);
                    this.isPlaying = false;
                    this.currentAudio = null;
                    this.updatePlayingState(false);
                    this.playNextInQueue();
                    resolve();
                };
                
                audio.onerror = (e) => {
                    console.error('❌ [Voice] Audio playback error:', e);
                    console.error('❌ [Voice] Audio error details:', audio.error);
                    URL.revokeObjectURL(audioUrl);
                    this.isPlaying = false;
                    this.currentAudio = null;
                    this.updatePlayingState(false);
                    reject(new Error('Audio playback failed: ' + (audio.error ? audio.error.message : 'Unknown')));
                };
                
                console.log('▶️ [Voice] Attempting to play audio...');
                
                audio.play()
                    .then(() => {
                        console.log('✅ [Voice] Audio playback started successfully');
                    })
                    .catch(err => {
                        console.warn('⚠️ [Voice] Autoplay blocked by browser:', err);
                        console.warn('⚠️ [Voice] User interaction required for audio playback');
                        // Some browsers block autoplay - notify user
                        if (this.shell && this.shell.toast) {
                            this.shell.toast('🔊 Click to enable audio playback', 'info');
                        }
                        this.isPlaying = false;
                        this.currentAudio = null;
                        this.updatePlayingState(false);
                        resolve();
                    });
                
            } catch (error) {
                console.error('❌ [Voice] playAudio exception:', error);
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
