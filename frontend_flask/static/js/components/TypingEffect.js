// components/TypingEffect.js - Handles the typewriter animation for the agent

export class TypingEffect {
    constructor(element, text, onComplete) {
        this.element = element;
        this.text = text;
        this.onComplete = onComplete;
        this.cursorElement = document.createElement('span');
        this.cursorElement.className = 'typing-cursor';
        this.element.appendChild(this.cursorElement);
        this.stop = false;
    }

    start() {
        let i = 0;
        const speed = 20; // ms per char

        const typeChar = () => {
            if (this.stop) return;

            if (i < this.text.length) {
                // Insert before cursor
                this.element.insertBefore(document.createTextNode(this.text.charAt(i)), this.cursorElement);
                i++;

                // Randomize speed slightly for realism
                const randomSpeed = speed + (Math.random() * 20 - 10);
                setTimeout(typeChar, randomSpeed);
            } else {
                // Done
                this.cursorElement.remove();
                if (this.onComplete) this.onComplete();
            }
        };

        typeChar();
    }

    cancel() {
        this.stop = true;
        this.element.textContent = this.text; // Show full text immediately
    }
}
