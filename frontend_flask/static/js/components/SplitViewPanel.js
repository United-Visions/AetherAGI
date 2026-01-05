// components/SplitViewPanel.js - Split view for detailed task inspection

export class SplitViewPanel {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.isOpen = false;
        this.currentActivity = null;
        
        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="split-view-panel" id="split-view-panel">
                <div class="split-view-header">
                    <div class="split-view-title">
                        <i class="fas fa-layer-group"></i>
                        <span id="split-view-title-text">Task Details</span>
                    </div>
                    <button class="split-view-close" id="split-view-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                
                <div class="split-view-tabs">
                    <button class="split-view-tab active" data-tab="overview">
                        <i class="fas fa-info-circle"></i> Overview
                    </button>
                    <button class="split-view-tab" data-tab="code">
                        <i class="fas fa-code"></i> Code
                    </button>
                    <button class="split-view-tab" data-tab="diff">
                        <i class="fas fa-exchange-alt"></i> Changes
                    </button>
                    <button class="split-view-tab" data-tab="preview">
                        <i class="fas fa-eye"></i> Preview
                    </button>
                    <button class="split-view-tab" data-tab="environment">
                        <i class="fas fa-server"></i> Environment
                    </button>
                </div>
                
                <div class="split-view-content" id="split-view-content">
                    <!-- Content injected based on tab -->
                </div>
            </div>
        `;

        // Event listeners
        document.getElementById('split-view-close').addEventListener('click', () => this.close());
        
        document.querySelectorAll('.split-view-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.currentTarget.dataset.tab;
                this.switchTab(tabName);
            });
        });

        // Listen for activity selections
        document.addEventListener('activity-selected', (e) => {
            this.open(e.detail);
        });
    }

    open(activity) {
        this.currentActivity = activity;
        this.isOpen = true;
        
        const panel = document.getElementById('split-view-panel');
        panel.classList.add('open');
        
        document.getElementById('split-view-title-text').textContent = activity.title;
        
        this.switchTab('overview');
    }

    close() {
        this.isOpen = false;
        const panel = document.getElementById('split-view-panel');
        panel.classList.remove('open');
    }

    switchTab(tabName) {
        // Update active tab
        document.querySelectorAll('.split-view-tab').forEach(tab => {
            tab.classList.remove('active');
            if (tab.dataset.tab === tabName) {
                tab.classList.add('active');
            }
        });

        // Render content
        const content = document.getElementById('split-view-content');
        
        switch(tabName) {
            case 'overview':
                content.innerHTML = this.renderOverview();
                break;
            case 'code':
                content.innerHTML = this.renderCode();
                this.initCodeHighlighting();
                break;
            case 'diff':
                content.innerHTML = this.renderDiff();
                break;
            case 'preview':
                content.innerHTML = this.renderPreview();
                break;
            case 'environment':
                content.innerHTML = this.renderEnvironment();
                break;
        }
    }

    renderOverview() {
        if (!this.currentActivity) return '<div class="empty-state">No activity selected</div>';
        
        const activity = this.currentActivity;
        
        return `
            <div class="overview-container">
                <div class="overview-section">
                    <h3><i class="fas fa-info-circle"></i> Activity Information</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">Type:</span>
                            <span class="info-value">${activity.type}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Status:</span>
                            <span class="info-value status-${activity.status}">${activity.status}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Started:</span>
                            <span class="info-value">${new Date(activity.timestamp).toLocaleString()}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Duration:</span>
                            <span class="info-value">${this.calculateDuration(activity)}</span>
                        </div>
                    </div>
                </div>
                
                <div class="overview-section">
                    <h3><i class="fas fa-align-left"></i> Details</h3>
                    <p class="details-text">${activity.details || 'No additional details available.'}</p>
                </div>
                
                ${activity.data?.reasoning ? `
                    <div class="overview-section">
                        <h3><i class="fas fa-brain"></i> Agent Reasoning</h3>
                        <div class="reasoning-steps">
                            ${activity.data.reasoning.map((step, i) => `
                                <div class="reasoning-step">
                                    <div class="step-number">${i + 1}</div>
                                    <div class="step-content">${step}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${activity.data?.files ? `
                    <div class="overview-section">
                        <h3><i class="fas fa-file-code"></i> Files Affected</h3>
                        <ul class="file-list">
                            ${activity.data.files.map(file => `
                                <li>
                                    <i class="fas fa-file"></i>
                                    <span>${file}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
    }

    renderCode() {
        if (!this.currentActivity?.data?.code) {
            return '<div class="empty-state">No code available for this activity</div>';
        }

        const code = this.currentActivity.data.code;
        const language = this.currentActivity.data.language || 'python';
        const files = this.currentActivity.data.files || [];

        return `
            <div class="code-viewer">
                <div class="code-header">
                    <span class="code-language">${language}</span>
                    ${files.length > 0 ? `<span class="code-files">${files.join(', ')}</span>` : ''}
                    <button class="copy-code-btn" onclick="navigator.clipboard.writeText(\`${code.replace(/`/g, '\\`')}\`); this.innerHTML='<i class=\\"fas fa-check\\"></i> Copied';">
                        <i class="fas fa-copy"></i> Copy
                    </button>
                </div>
                <pre><code class="language-${language}" id="code-content">${this.escapeHtml(code)}</code></pre>
            </div>
        `;
    }

    renderDiff() {
        if (!this.currentActivity?.data?.diff) {
            return '<div class="empty-state">No changes to display</div>';
        }

        const diff = this.currentActivity.data.diff;
        
        return `
            <div class="diff-viewer">
                <div class="diff-header">
                    <span class="diff-stats">
                        <span class="additions">+${diff.additions || 0}</span>
                        <span class="deletions">-${diff.deletions || 0}</span>
                    </span>
                </div>
                <div class="diff-content">
                    ${this.renderDiffLines(diff.changes)}
                </div>
            </div>
        `;
    }

    renderDiffLines(changes) {
        if (!changes || !Array.isArray(changes)) return '';
        
        return changes.map(change => {
            const type = change.type; // 'add', 'remove', 'context'
            const lineClass = `diff-line diff-${type}`;
            const prefix = type === 'add' ? '+' : type === 'remove' ? '-' : ' ';
            
            return `
                <div class="${lineClass}">
                    <span class="line-number">${change.lineNumber || ''}</span>
                    <span class="line-prefix">${prefix}</span>
                    <span class="line-content">${this.escapeHtml(change.content)}</span>
                </div>
            `;
        }).join('');
    }

    renderPreview() {
        if (!this.currentActivity?.data?.preview_url && !this.currentActivity?.data?.preview_html) {
            return '<div class="empty-state"><i class="fas fa-eye-slash"></i><br>No preview available</div>';
        }

        if (this.currentActivity.data.preview_url) {
            return `
                <div class="preview-container">
                    <div class="preview-header">
                        <span>Live Preview</span>
                        <a href="${this.currentActivity.data.preview_url}" target="_blank" class="external-link">
                            <i class="fas fa-external-link-alt"></i> Open in new tab
                        </a>
                    </div>
                    <iframe 
                        src="${this.currentActivity.data.preview_url}" 
                        class="preview-iframe"
                        sandbox="allow-scripts allow-same-origin"
                    ></iframe>
                </div>
            `;
        }

        return `
            <div class="preview-container">
                <div class="preview-header">
                    <span>Preview</span>
                </div>
                <div class="preview-html">
                    ${this.currentActivity.data.preview_html}
                </div>
            </div>
        `;
    }

    renderEnvironment() {
        const env = this.currentActivity?.data?.environment || {};
        
        return `
            <div class="environment-viewer">
                <div class="env-section">
                    <h3><i class="fas fa-cube"></i> Runtime Environment</h3>
                    <div class="env-grid">
                        <div class="env-item">
                            <span class="env-label">Python Version:</span>
                            <span class="env-value">${env.python_version || '3.11.0'}</span>
                        </div>
                        <div class="env-item">
                            <span class="env-label">Working Directory:</span>
                            <span class="env-value">${env.working_dir || '/workspace'}</span>
                        </div>
                        <div class="env-item">
                            <span class="env-label">Execution Time:</span>
                            <span class="env-value">${env.execution_time || '0.5s'}</span>
                        </div>
                    </div>
                </div>
                
                <div class="env-section">
                    <h3><i class="fas fa-box"></i> Dependencies</h3>
                    <ul class="dependency-list">
                        ${(env.dependencies || ['requests', 'beautifulsoup4', 'pandas']).map(dep => `
                            <li>
                                <i class="fas fa-cube"></i>
                                <span>${dep}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                
                ${env.logs ? `
                    <div class="env-section">
                        <h3><i class="fas fa-terminal"></i> Console Output</h3>
                        <pre class="console-output">${this.escapeHtml(env.logs)}</pre>
                    </div>
                ` : ''}
            </div>
        `;
    }

    calculateDuration(activity) {
        const start = new Date(activity.timestamp);
        const end = activity.completed_at ? new Date(activity.completed_at) : new Date();
        const diff = end - start;
        
        if (diff < 1000) return `${diff}ms`;
        if (diff < 60000) return `${(diff / 1000).toFixed(1)}s`;
        return `${(diff / 60000).toFixed(1)}m`;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    initCodeHighlighting() {
        // If you have a syntax highlighting library like Prism.js or Highlight.js
        // You can initialize it here
        const codeBlock = document.getElementById('code-content');
        if (codeBlock && window.Prism) {
            window.Prism.highlightElement(codeBlock);
        }
    }
}
