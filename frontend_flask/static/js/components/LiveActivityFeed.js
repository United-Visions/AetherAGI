// components/LiveActivityFeed.js - Real-time AGI Activity Tracking

export class LiveActivityFeed {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.activities = [];
        this.maxActivities = 5;
        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="activity-feed-header">
                <div class="activity-feed-title">
                    <i class="fas fa-bolt"></i>
                    <span>AGI Activity Stream</span>
                </div>
                <button class="activity-feed-expand" title="View All">
                    <i class="fas fa-expand"></i>
                </button>
            </div>
            <div class="activity-feed-items"></div>
        `;

        this.itemsContainer = this.container.querySelector('.activity-feed-items');
        this.setupExpandButton();
    }

    setupExpandButton() {
        const expandBtn = this.container.querySelector('.activity-feed-expand');
        expandBtn.addEventListener('click', () => {
            this.openDetailedView();
        });
    }

    addActivity(type, title, details = {}) {
        const activity = {
            id: Date.now() + Math.random(),
            type,
            title,
            details,
            timestamp: new Date(),
            status: 'active'
        };

        this.activities.unshift(activity);
        if (this.activities.length > 50) {
            this.activities = this.activities.slice(0, 50);
        }

        this.renderActivity(activity);
        return activity.id;
    }

    renderActivity(activity) {
        const item = document.createElement('div');
        item.className = `activity-item activity-${activity.type} activity-${activity.status}`;
        item.dataset.activityId = activity.id;

        const icon = this.getActivityIcon(activity.type);
        const color = this.getActivityColor(activity.type);

        item.innerHTML = `
            <div class="activity-indicator" style="background-color: ${color}">
                <i class="${icon}"></i>
            </div>
            <div class="activity-content">
                <div class="activity-title">${activity.title}</div>
                <div class="activity-time">${this.formatTime(activity.timestamp)}</div>
            </div>
            <div class="activity-status">
                <div class="activity-spinner"></div>
            </div>
        `;

        item.addEventListener('click', () => {
            this.showActivityDetails(activity);
        });

        // Slide in animation
        item.style.transform = 'translateX(-100%)';
        item.style.opacity = '0';
        
        this.itemsContainer.insertBefore(item, this.itemsContainer.firstChild);

        setTimeout(() => {
            item.style.transition = 'all 0.3s ease';
            item.style.transform = 'translateX(0)';
            item.style.opacity = '1';
        }, 10);

        // Remove old items if too many
        const items = this.itemsContainer.querySelectorAll('.activity-item');
        if (items.length > this.maxActivities) {
            const lastItem = items[items.length - 1];
            lastItem.style.opacity = '0';
            lastItem.style.transform = 'translateX(100%)';
            setTimeout(() => lastItem.remove(), 300);
        }
    }

    updateActivity(activityId, status, message = null) {
        const activity = this.activities.find(a => a.id === activityId);
        if (!activity) return;

        activity.status = status;
        if (message) activity.statusMessage = message;

        const item = this.itemsContainer.querySelector(`[data-activity-id="${activityId}"]`);
        if (!item) return;

        item.className = `activity-item activity-${activity.type} activity-${status}`;
        
        const statusEl = item.querySelector('.activity-status');
        if (status === 'complete') {
            statusEl.innerHTML = '<i class="fas fa-check-circle"></i>';
        } else if (status === 'error') {
            statusEl.innerHTML = '<i class="fas fa-exclamation-circle"></i>';
        }
    }

    showActivityDetails(activity) {
        const modal = document.createElement('div');
        modal.className = 'activity-modal-overlay';
        modal.innerHTML = `
            <div class="activity-modal">
                <div class="activity-modal-header">
                    <h3>
                        <i class="${this.getActivityIcon(activity.type)}"></i>
                        ${activity.title}
                    </h3>
                    <button class="activity-modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="activity-modal-content">
                    ${this.renderActivityDetails(activity)}
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        modal.querySelector('.activity-modal-close').addEventListener('click', () => {
            modal.remove();
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
    }

    renderActivityDetails(activity) {
        let html = `
            <div class="activity-detail-section">
                <h4>Status</h4>
                <p class="activity-status-badge status-${activity.status}">
                    ${activity.status.toUpperCase()}
                </p>
            </div>
            <div class="activity-detail-section">
                <h4>Started</h4>
                <p>${activity.timestamp.toLocaleString()}</p>
            </div>
        `;

        if (activity.details) {
            if (activity.details.files) {
                html += `
                    <div class="activity-detail-section">
                        <h4>Files Modified</h4>
                        <ul class="activity-file-list">
                            ${activity.details.files.map(f => `
                                <li>
                                    <i class="fas fa-file-code"></i>
                                    ${f}
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                `;
            }

            if (activity.details.code) {
                html += `
                    <div class="activity-detail-section">
                        <h4>Code Preview</h4>
                        <pre><code class="language-${activity.details.language || 'python'}">${this.escapeHtml(activity.details.code)}</code></pre>
                    </div>
                `;
            }

            if (activity.details.logs) {
                html += `
                    <div class="activity-detail-section">
                        <h4>Execution Logs</h4>
                        <div class="activity-logs">
                            ${activity.details.logs.map(log => `
                                <div class="log-line log-${log.level}">
                                    <span class="log-time">${log.time}</span>
                                    <span class="log-message">${log.message}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            }
        }

        return html;
    }

    openDetailedView() {
        const modal = document.createElement('div');
        modal.className = 'activity-modal-overlay';
        modal.innerHTML = `
            <div class="activity-modal activity-modal-large">
                <div class="activity-modal-header">
                    <h3>
                        <i class="fas fa-history"></i>
                        Complete Activity History
                    </h3>
                    <button class="activity-modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="activity-modal-content">
                    <div class="activity-timeline">
                        ${this.activities.map(activity => `
                            <div class="timeline-item">
                                <div class="timeline-marker" style="background-color: ${this.getActivityColor(activity.type)}">
                                    <i class="${this.getActivityIcon(activity.type)}"></i>
                                </div>
                                <div class="timeline-content">
                                    <div class="timeline-title">${activity.title}</div>
                                    <div class="timeline-time">${activity.timestamp.toLocaleString()}</div>
                                    <div class="timeline-status status-${activity.status}">${activity.status}</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        modal.querySelector('.activity-modal-close').addEventListener('click', () => {
            modal.remove();
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
    }

    getActivityIcon(type) {
        const icons = {
            'tool_create': 'fas fa-wrench',
            'memory_update': 'fas fa-brain',
            'research': 'fas fa-search',
            'learning': 'fas fa-graduation-cap',
            'self_modify': 'fas fa-code-branch',
            'knowledge_query': 'fas fa-database',
            'safety_check': 'fas fa-shield-alt',
            'tool_execute': 'fas fa-play-circle',
            'file_process': 'fas fa-file-alt',
            'conversation': 'fas fa-comments'
        };
        return icons[type] || 'fas fa-circle';
    }

    getActivityColor(type) {
        const colors = {
            'tool_create': '#8b5cf6',
            'memory_update': '#10b981',
            'research': '#3b82f6',
            'learning': '#f59e0b',
            'self_modify': '#ef4444',
            'knowledge_query': '#06b6d4',
            'safety_check': '#ec4899',
            'tool_execute': '#10b981',
            'file_process': '#6366f1',
            'conversation': '#64748b'
        };
        return colors[type] || '#6b7280';
    }

    formatTime(date) {
        const now = new Date();
        const diff = now - date;
        const seconds = Math.floor(diff / 1000);
        
        if (seconds < 60) return `${seconds}s ago`;
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;
        return date.toLocaleDateString();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
