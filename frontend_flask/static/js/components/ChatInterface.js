// components/ChatInterface.js - Handles message rendering

import { TypingEffect } from './TypingEffect.js';

export class ChatInterface {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }

    addMessage(role, content, metadata = {}) {
        const row = document.createElement('div');
        row.className = `message-row ${role === 'user' ? 'user' : 'assistant'}`;

        if (role === 'user') {
            const bubble = document.createElement('div');
            bubble.className = 'user-bubble';
            bubble.innerText = content;
            row.appendChild(bubble);
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

            // Handle Emoji/Status if metadata exists
            if (metadata.surprise > 0.5 || metadata.research) {
                this.addSentimentEmoji(row, metadata);
            }

            // Typing effect for assistant
            new TypingEffect(textDiv, content, () => {
                this.scrollToBottom();
            }).start();
        }

        this.container.appendChild(row);
        this.scrollToBottom();
    }

    addSentimentEmoji(rowElement, metadata) {
        const bubble = rowElement.querySelector('.assistant-text') || rowElement.querySelector('.user-bubble');
        if (!bubble) return;

        let emoji = '😐'; // Default
        let text = 'Processing...';

        if (metadata.research) {
            emoji = '🧪';
            text = "I'm conducting research on this.";
        } else if (metadata.surprise > 0.7) {
            emoji = '😲';
            text = `This is highly surprising! (Score: ${metadata.surprise.toFixed(2)})`;
        } else if (metadata.surprise > 0.5) {
            emoji = '🤔';
            text = `This is interesting. (Score: ${metadata.surprise.toFixed(2)})`;
        }

        const badge = document.createElement('div');
        badge.className = 'message-status';
        badge.innerText = emoji;

        const tooltip = document.createElement('div');
        tooltip.className = 'status-tooltip';
        tooltip.innerText = text;

        badge.appendChild(tooltip);

        // Append to the wrapper if assistant, or bubble if user (though prompt said agent feelings on user bubble? "allow it to show the emgie in the top right of the users chat bubble and the agents message")
        // Implementation: Add to the specific message row passed in.
        // Note: CSS expects .message-status to be absolute relative to a parent.
        // We need to ensure the parent has position: relative.
        // .user-bubble and .assistant-text have position:relative in CSS.

        bubble.appendChild(badge);
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
