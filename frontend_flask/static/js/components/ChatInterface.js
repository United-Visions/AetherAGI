// components/ChatInterface.js - Handles message rendering

import { TypingEffect } from './TypingEffect.js';

export class ChatInterface {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        // Global listener to close tooltips when clicking outside
        document.addEventListener('click', (e) => {
            // If the click is not on a badge, close all tooltips
            if (!e.target.closest('.message-status')) {
                document.querySelectorAll('.status-tooltip').forEach(t => t.style.display = 'none');
            }
        });
    }

    addMessage(role, content, metadata = {}) {
        const row = document.createElement('div');
        row.className = `message-row ${role === 'user' ? 'user' : 'assistant'}`;

        if (role === 'user') {
            const bubble = document.createElement('div');
            bubble.className = 'user-bubble';
            bubble.innerText = content;
            row.appendChild(bubble);

            // Check for user emotions (Top Right)
            if (metadata.user_emotion) {
                this.addUserEmotionEmoji(row, metadata.user_emotion);
            }

        } else {
            // Assistant layout: Avatar + Text
            // We create a wrapper to align avatar and text
            const contentWrapper = document.createElement('div');
            contentWrapper.style.display = 'flex';
            contentWrapper.style.alignItems = 'flex-start';

            const avatar = document.createElement('div');
            avatar.className = 'agent-avatar-icon';
            avatar.innerHTML = '<i class="fas fa-robot"></i>';

            const textDiv = document.createElement('div');
            textDiv.className = 'assistant-text';

            contentWrapper.appendChild(avatar);
            contentWrapper.appendChild(textDiv);
            row.appendChild(contentWrapper);

            // Check for agent state (Top Left)
            if (metadata.agent_state) {
                 this.addAgentStateEmoji(row, metadata.agent_state);
            }
            // Backward compatibility for old logic if needed, or simple direct check
            else if (metadata.surprise > 0.5 || metadata.research) {
                 this.addAgentStateEmoji(row, {surprise_score: metadata.surprise, is_researching: metadata.research});
            }

            // Typing effect for assistant
            new TypingEffect(textDiv, content, () => {
                this.scrollToBottom();
            }).start();
        }

        this.container.appendChild(row);
        this.scrollToBottom();
    }

    addUserEmotionEmoji(rowElement, emotionData) {
        const bubble = rowElement.querySelector('.user-bubble');
        if (!bubble) return;

        let emoji = '';
        let text = '';

        // Logic based on Valence
        if (emotionData.valence > 0.5) {
            emoji = '😊';
            text = "I sense you are feeling positive.";
        } else if (emotionData.valence < -0.5) {
            emoji = '😔';
            text = "I sense you might be frustrated or unhappy.";
        } else if (emotionData.moral_sentiment && emotionData.moral_sentiment > 0.3) {
             // Example for curiosity if we had it, or use keyword detection from backend if exposed
             // Backend 'curious' keyword sets valence=0.1, arousal=0.4.
             // Let's use arousal/valence combo for 'Curious' approximation
             if (emotionData.valence > 0 && emotionData.arousal > 0.3) {
                 emoji = '🤔';
                 text = "I sense you are curious.";
             }
        }

        if (!emoji) return; // Don't show if neutral

        this.renderBadge(bubble, emoji, text, 'top-right');
    }

    addAgentStateEmoji(rowElement, agentState) {
        const bubble = rowElement.querySelector('.assistant-text');
        if (!bubble) return;

        let emoji = '';
        let text = '';

        if (agentState.is_researching) {
            emoji = '🧪';
            text = "I'm conducting research to learn more about this.";
        } else if (agentState.surprise_score > 0.7) {
            emoji = '😲';
            text = `This is highly surprising! (Score: ${agentState.surprise_score.toFixed(2)})`;
        } else if (agentState.surprise_score > 0.5) {
            emoji = '🧐';
            text = `This is interesting. (Score: ${agentState.surprise_score.toFixed(2)})`;
        }

        if (!emoji) return;

        this.renderBadge(bubble, emoji, text, 'top-left');
    }

    renderBadge(parentElement, emoji, tooltipText, positionClass) {
        const badge = document.createElement('div');
        badge.className = `message-status status-${positionClass}`;
        badge.innerText = emoji;

        const tooltip = document.createElement('div');
        tooltip.className = 'status-tooltip';
        tooltip.innerText = tooltipText;

        // Toggle visibility on click
        badge.addEventListener('click', (e) => {
            // Stop propagation so the global listener doesn't immediately close it
            e.stopPropagation();

            const isVisible = tooltip.style.display === 'block';

            // Close all other tooltips first
            document.querySelectorAll('.status-tooltip').forEach(t => t.style.display = 'none');

            // Toggle this one
            tooltip.style.display = isVisible ? 'none' : 'block';
        });

        badge.appendChild(tooltip);
        parentElement.appendChild(badge);
    }

    addFileAttachmentMessage(role, fileName, fileType) {
        const row = document.createElement('div');
        row.className = `message-row ${role === 'user' ? 'user' : 'assistant'}`;

        const bubble = document.createElement('div');
        bubble.className = role === 'user' ? 'user-bubble' : 'assistant-text';
        if (role === 'assistant') bubble.style.marginLeft = '32px'; // Offset for missing avatar

        const attachment = document.createElement('div');
        attachment.className = 'chat-attachment';

        let icon = 'fa-file';
        if (fileType.startsWith('image/')) icon = 'fa-image';
        if (fileType.startsWith('video/')) icon = 'fa-video';

        attachment.innerHTML = `<i class="fas ${icon}"></i> <span>${fileName}</span>`;

        bubble.appendChild(attachment);
        row.appendChild(bubble);
        this.container.appendChild(row);
        this.scrollToBottom();
    }

    scrollToBottom() {
        this.container.scrollTop = this.container.scrollHeight;
    }
}
