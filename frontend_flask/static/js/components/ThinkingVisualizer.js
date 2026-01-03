// components/ThinkingVisualizer.js - Handles the "Thinking" cards

export class ThinkingVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }

    createCard(title = "Thinking Process") {
        const card = document.createElement('div');
        card.className = 'thought-card';

        const header = document.createElement('div');
        header.className = 'thought-header';
        header.innerHTML = `
            <div class="thought-title">
                <i class="fas fa-brain"></i>
                <span>${title}</span>
            </div>
            <i class="fas fa-chevron-down"></i>
        `;

        const content = document.createElement('div');
        content.className = 'thought-content';
        content.innerText = "Initializing...";

        card.appendChild(header);
        card.appendChild(content);

        // Add click listener to toggle expand
        header.addEventListener('click', () => {
            card.classList.toggle('expanded');
        });

        this.container.appendChild(card);
        this.container.scrollTop = this.container.scrollHeight; // Auto scroll

        return {
            update: (text) => {
                content.innerText = text;
            },
            append: (text) => {
                content.innerText += "\n" + text;
            },
            setSuccess: () => {
                header.querySelector('.thought-title span').style.color = 'var(--success-color)';
            },
            setError: () => {
                header.querySelector('.thought-title span').style.color = 'var(--error-color)';
            },
            element: card
        };
    }

    clear() {
        this.container.innerHTML = '';
    }
}
