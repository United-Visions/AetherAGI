// components/StreamingAccordion.js - Advanced AGI Transparency System

export class StreamingAccordion {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.activePanels = new Set();
        this.messageId = null;
        this.init();
    }

    init() {
        this.container.innerHTML = '';
        this.container.className = 'streaming-accordion-container';
    }

    startNewCycle(messageId) {
        this.messageId = messageId;
        this.activePanels.clear();
        this.init();
    }

    createPanel(panelId, title, icon, priority = 0, initiallyExpanded = false) {
        if (this.activePanels.has(panelId)) {
            return document.getElementById(`panel-${panelId}`);
        }

        const panel = document.createElement('div');
        panel.className = `accordion-panel priority-${priority}`;
        panel.id = `panel-${panelId}`;
        panel.dataset.panelId = panelId;
        
        const header = document.createElement('div');
        header.className = 'accordion-header';
        header.innerHTML = `
            <div class="accordion-header-content">
                <i class="${icon} accordion-icon"></i>
                <span class="accordion-title">${title}</span>
                <span class="accordion-status">
                    <div class="pulse-dot"></div>
                    <span class="status-text">Processing...</span>
                </span>
            </div>
            <i class="fas fa-chevron-down accordion-toggle"></i>
        `;

        const content = document.createElement('div');
        content.className = `accordion-content ${initiallyExpanded ? 'expanded' : ''}`;

        const contentInner = document.createElement('div');
        contentInner.className = 'accordion-content-inner';
        content.appendChild(contentInner);

        panel.appendChild(header);
        panel.appendChild(content);

        // Toggle functionality
        header.addEventListener('click', () => {
            const isExpanded = content.classList.contains('expanded');
            content.classList.toggle('expanded');
            header.querySelector('.accordion-toggle').style.transform = 
                isExpanded ? 'rotate(0deg)' : 'rotate(180deg)';
        });

        // Animate panel entrance
        panel.style.opacity = '0';
        panel.style.transform = 'translateY(-10px)';
        this.container.appendChild(panel);
        
        setTimeout(() => {
            panel.style.transition = 'all 0.3s ease';
            panel.style.opacity = '1';
            panel.style.transform = 'translateY(0)';
        }, 50);

        this.activePanels.add(panelId);
        return panel;
    }

    streamToPanel(panelId, content, type = 'text') {
        const panel = document.getElementById(`panel-${panelId}`);
        if (!panel) return;

        const contentInner = panel.querySelector('.accordion-content-inner');
        
        switch(type) {
            case 'text':
                this.appendText(contentInner, content);
                break;
            case 'code':
                this.appendCode(contentInner, content);
                break;
            case 'metric':
                this.appendMetric(contentInner, content);
                break;
            case 'graph':
                this.appendGraph(contentInner, content);
                break;
            case 'list':
                this.appendList(contentInner, content);
                break;
            case 'diff':
                this.appendDiff(contentInner, content);
                break;
            case 'emotion':
                this.appendEmotion(contentInner, content);
                break;
        }

        // Auto-scroll to show new content
        contentInner.scrollTop = contentInner.scrollHeight;
    }

    appendText(container, text) {
        const line = document.createElement('div');
        line.className = 'stream-line fade-in';
        line.textContent = text;
        container.appendChild(line);
    }

    appendCode(container, codeData) {
        const codeBlock = document.createElement('pre');
        codeBlock.className = 'stream-code fade-in';
        const code = document.createElement('code');
        code.className = `language-${codeData.language || 'python'}`;
        code.textContent = codeData.content;
        codeBlock.appendChild(code);
        container.appendChild(codeBlock);
        
        // Syntax highlighting if available
        if (window.Prism) {
            Prism.highlightElement(code);
        }
    }

    appendMetric(container, metricData) {
        const metric = document.createElement('div');
        metric.className = 'stream-metric fade-in';
        metric.innerHTML = `
            <span class="metric-label">${metricData.label}:</span>
            <span class="metric-value ${metricData.status || ''}">${metricData.value}</span>
            ${metricData.unit ? `<span class="metric-unit">${metricData.unit}</span>` : ''}
        `;
        container.appendChild(metric);
    }

    appendGraph(container, graphData) {
        const graph = document.createElement('div');
        graph.className = 'stream-graph fade-in';
        
        if (graphData.type === 'bar') {
            this.renderBarGraph(graph, graphData);
        } else if (graphData.type === 'line') {
            this.renderLineGraph(graph, graphData);
        } else if (graphData.type === 'emotion') {
            this.renderEmotionGraph(graph, graphData);
        }
        
        container.appendChild(graph);
    }

    renderBarGraph(container, data) {
        const max = Math.max(...data.values);
        container.innerHTML = `
            <div class="bar-graph">
                ${data.labels.map((label, i) => `
                    <div class="bar-item">
                        <div class="bar-label">${label}</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: ${(data.values[i] / max) * 100}%">
                                <span class="bar-value">${data.values[i]}</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    renderLineGraph(container, data) {
        // Simple SVG line graph
        const width = 300;
        const height = 100;
        const padding = 20;
        const max = Math.max(...data.values);
        const min = Math.min(...data.values);
        const range = max - min || 1;
        
        const points = data.values.map((val, i) => {
            const x = padding + (i / (data.values.length - 1)) * (width - 2 * padding);
            const y = height - padding - ((val - min) / range) * (height - 2 * padding);
            return `${x},${y}`;
        }).join(' ');

        container.innerHTML = `
            <svg class="line-graph" viewBox="0 0 ${width} ${height}">
                <polyline points="${points}" fill="none" stroke="#10b981" stroke-width="2"/>
                ${data.values.map((val, i) => {
                    const x = padding + (i / (data.values.length - 1)) * (width - 2 * padding);
                    const y = height - padding - ((val - min) / range) * (height - 2 * padding);
                    return `<circle cx="${x}" cy="${y}" r="3" fill="#10b981"/>`;
                }).join('')}
            </svg>
        `;
    }

    renderEmotionGraph(container, data) {
        container.innerHTML = `
            <div class="emotion-graph">
                <div class="emotion-axis">
                    <div class="axis-label">Valence</div>
                    <div class="emotion-bar">
                        <div class="emotion-fill valence" style="width: ${(data.valence + 1) * 50}%"></div>
                        <div class="emotion-marker" style="left: 50%"></div>
                    </div>
                    <div class="axis-values">
                        <span>Negative</span>
                        <span>Neutral</span>
                        <span>Positive</span>
                    </div>
                </div>
                <div class="emotion-axis">
                    <div class="axis-label">Arousal</div>
                    <div class="emotion-bar">
                        <div class="emotion-fill arousal" style="width: ${(data.arousal + 1) * 50}%"></div>
                        <div class="emotion-marker" style="left: 50%"></div>
                    </div>
                    <div class="axis-values">
                        <span>Calm</span>
                        <span>Neutral</span>
                        <span>Excited</span>
                    </div>
                </div>
                ${data.moral_sentiment !== undefined ? `
                <div class="emotion-axis">
                    <div class="axis-label">Moral Sentiment</div>
                    <div class="emotion-bar">
                        <div class="emotion-fill moral" style="width: ${data.moral_sentiment * 100}%"></div>
                    </div>
                    <div class="axis-values">
                        <span>Low</span>
                        <span>High</span>
                    </div>
                </div>
                ` : ''}
            </div>
        `;
    }

    appendList(container, listData) {
        const list = document.createElement('ul');
        list.className = 'stream-list fade-in';
        listData.items.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `
                <i class="${item.icon || 'fas fa-circle'} list-icon"></i>
                <span>${item.text}</span>
                ${item.badge ? `<span class="list-badge ${item.badge.type}">${item.badge.text}</span>` : ''}
            `;
            list.appendChild(li);
        });
        container.appendChild(list);
    }

    appendDiff(container, diffData) {
        const diff = document.createElement('div');
        diff.className = 'stream-diff fade-in';
        diff.innerHTML = `
            <div class="diff-header">
                <span class="diff-file">${diffData.file}</span>
                <span class="diff-stats">
                    <span class="diff-added">+${diffData.added}</span>
                    <span class="diff-removed">-${diffData.removed}</span>
                </span>
            </div>
            <div class="diff-content">
                ${diffData.changes.map(change => `
                    <div class="diff-line ${change.type}">
                        <span class="diff-marker">${change.type === 'added' ? '+' : change.type === 'removed' ? '-' : ' '}</span>
                        <span class="diff-text">${this.escapeHtml(change.content)}</span>
                    </div>
                `).join('')}
            </div>
        `;
        container.appendChild(diff);
    }

    appendEmotion(container, emotionData) {
        const emotion = document.createElement('div');
        emotion.className = 'stream-emotion fade-in';
        
        const emoji = this.getEmotionEmoji(emotionData);
        const description = this.getEmotionDescription(emotionData);
        
        emotion.innerHTML = `
            <div class="emotion-display">
                <div class="emotion-emoji">${emoji}</div>
                <div class="emotion-details">
                    <div class="emotion-label">${description}</div>
                    <div class="emotion-confidence">Confidence: ${(emotionData.confidence * 100).toFixed(0)}%</div>
                </div>
            </div>
        `;
        container.appendChild(emotion);
    }

    getEmotionEmoji(data) {
        if (data.valence > 0.5 && data.arousal > 0.5) return '😄';
        if (data.valence > 0.5 && data.arousal < -0.5) return '😊';
        if (data.valence < -0.5 && data.arousal > 0.5) return '😠';
        if (data.valence < -0.5 && data.arousal < -0.5) return '😔';
        if (data.arousal > 0.5) return '🤔';
        return '😐';
    }

    getEmotionDescription(data) {
        if (data.valence > 0.5 && data.arousal > 0.5) return 'Excited & Positive';
        if (data.valence > 0.5 && data.arousal < -0.5) return 'Calm & Content';
        if (data.valence < -0.5 && data.arousal > 0.5) return 'Frustrated';
        if (data.valence < -0.5 && data.arousal < -0.5) return 'Sad or Disappointed';
        if (data.arousal > 0.5) return 'Curious';
        return 'Neutral';
    }

    setPanelStatus(panelId, status, message) {
        const panel = document.getElementById(`panel-${panelId}`);
        if (!panel) return;

        const statusElement = panel.querySelector('.accordion-status');
        const statusText = panel.querySelector('.status-text');
        const pulseDot = panel.querySelector('.pulse-dot');

        statusText.textContent = message;
        pulseDot.className = `pulse-dot ${status}`;
        
        if (status === 'success') {
            panel.classList.add('panel-complete');
            setTimeout(() => {
                if (!panel.querySelector('.accordion-content').classList.contains('expanded')) {
                    panel.style.opacity = '0.7';
                }
            }, 500);
        } else if (status === 'error') {
            panel.classList.add('panel-error');
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    clear() {
        this.init();
    }
}
