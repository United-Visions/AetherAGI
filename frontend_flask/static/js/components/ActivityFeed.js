// components/ActivityFeed.js - Real-time Activity Feed showing what Aether is doing

export class ActivityFeed {
    constructor(containerId) {
        console.log('🏗️ [ActivityFeed] Constructor called with containerId:', containerId);
        this.container = document.getElementById(containerId);
        
        if (!this.container) {
            console.error('❌ [ActivityFeed] Container not found:', containerId);
            return;
        }
        
        this.activities = [];
        this.maxActivities = 20;
        this.eventSource = null;
        console.log('✅ [ActivityFeed] Properties initialized');
        
        this.init();
    }

    init() {
        console.log('🚀 [ActivityFeed] Initializing activity feed UI...');
        this.container.innerHTML = `
            <div class="activity-feed-wrapper">
                <div class="activity-feed-header">
                    <i class="fas fa-brain"></i>
                    <span>Agent Activity Stream</span>
                    <div class="activity-pulse"></div>
                </div>
                <div class="activity-feed-scroll" id="activity-feed-scroll">
                    <!-- Activities injected here -->
                </div>
            </div>
        `;
        console.log('✅ [ActivityFeed] UI initialized');
    }

    addActivity(activity) {
        console.log('➕ [ActivityFeed] Adding activity:', activity.id, activity.type, activity.status);
        console.log('📋 [ActivityFeed] Activity details:', activity);
        /*
        activity: {
            id: unique_id,
            type: 'thinking' | 'tool_creation' | 'research' | 'memory_update' | 'file_change' | 'self_modification' | 'planning',
            status: 'pending' | 'in_progress' | 'completed' | 'error',
            title: short description,
            details: full details,
            timestamp: ISO string,
            data: {
                // Type-specific data
                files: [...],
                diff: {...},
                tool_code: "...",
                preview_url: "...",
                etc.
            }
        }
        */
        
        this.activities.unshift(activity);
        console.log('📊 [ActivityFeed] Total activities:', this.activities.length);
        
        if (this.activities.length > this.maxActivities) {
            const removed = this.activities.pop();
            console.log('🗑️ [ActivityFeed] Removed old activity:', removed.id);
        }
        
        this.render();
        console.log('✅ [ActivityFeed] Activity added and rendered');
        return activity.id;
    }

    updateActivity(id, updates) {
        console.log('🔄 [ActivityFeed] Updating activity:', id, 'Updates:', updates);
        const activity = this.activities.find(a => a.id === id);
        if (activity) {
            console.log('📝 [ActivityFeed] Current activity state:', activity);
            Object.assign(activity, updates);
            console.log('📝 [ActivityFeed] Updated activity state:', activity);
            this.render();
            console.log('✅ [ActivityFeed] Activity updated and re-rendered');
        } else {
            console.warn('⚠️ [ActivityFeed] Activity not found for update:', id);
        }
    }

    render() {
        console.log('🎨 [ActivityFeed] Rendering activities... Count:', this.activities.length);
        const scrollContainer = document.getElementById('activity-feed-scroll');
        if (!scrollContainer) {
            console.error('❌ [ActivityFeed] Scroll container not found!');
            return;
        }

        scrollContainer.innerHTML = this.activities.map(activity => {
            const icon = this.getIcon(activity.type);
            const statusClass = `activity-status-${activity.status}`;
            const time = this.formatTime(activity.timestamp);
            
            // Show code preview for file_change and code_execution types
            const hasCode = activity.data?.code && (activity.type === 'file_change' || activity.type === 'code_execution' || activity.type === 'tool_creation');
            const codePreview = hasCode ? `<div class="activity-code-preview">${this.getCodePreview(activity.data.code, activity.data.language)}</div>` : '';
            const filesInfo = activity.data?.files?.length > 0 ? `<div class="activity-files">📁 ${activity.data.files.join(', ')}</div>` : '';

            return `
                <div class="activity-card ${statusClass}" data-activity-id="${activity.id}">
                    <div class="activity-icon ${activity.type}">
                        <i class="${icon}"></i>
                        ${activity.status === 'in_progress' ? '<div class="spinner-ring"></div>' : ''}
                    </div>
                    <div class="activity-content">
                        <div class="activity-title">${activity.title}</div>
                        ${filesInfo}
                        ${codePreview}
                        <div class="activity-time">${time}</div>
                    </div>
                    <div class="activity-badge">
                        ${this.getStatusBadge(activity.status)}
                    </div>
                </div>
            `;
        }).join('');

        // Add click handlers
        scrollContainer.querySelectorAll('.activity-card').forEach(card => {
            card.addEventListener('click', () => {
                const id = card.dataset.activityId;
                const activity = this.activities.find(a => a.id === id);
                if (activity) {
                    this.onActivityClick(activity);
                }
            });
        });
    }
    
    getCodePreview(code, language = 'python') {
        if (!code) return '';
        // Show first 2 lines as preview
        const lines = code.split('\n').slice(0, 2);
        const preview = lines.join('\n');
        const truncated = code.split('\n').length > 2 ? '...' : '';
        return `<pre class="code-mini"><code class="language-${language}">${this.escapeHtml(preview)}${truncated}</code></pre>`;
    }

    getIcon(type) {
        const icons = {
            'thinking': 'fas fa-brain',
            'tool_creation': 'fas fa-tools',
            'research': 'fas fa-search',
            'memory_update': 'fas fa-database',
            'file_change': 'fas fa-file-code',
            'self_modification': 'fas fa-cogs',
            'planning': 'fas fa-project-diagram',
            'code_execution': 'fas fa-play-circle',
            'web_scraping': 'fas fa-spider',
            'api_call': 'fas fa-plug',
            'learning': 'fas fa-graduation-cap',
            'emotion_analysis': 'fas fa-heart',
            'surprise_detected': 'fas fa-exclamation-circle'
        };
        return icons[type] || 'fas fa-circle';
    }

    getStatusBadge(status) {
        const badges = {
            'pending': '<i class="fas fa-clock"></i>',
            'in_progress': '<i class="fas fa-spinner fa-spin"></i>',
            'completed': '<i class="fas fa-check-circle"></i>',
            'error': '<i class="fas fa-times-circle"></i>'
        };
        return badges[status] || '';
    }

    formatTime(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;

        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        return date.toLocaleTimeString();
    }

    onActivityClick(activity) {
        // Emit custom event for other components to handle
        const event = new CustomEvent('activity-selected', { detail: activity });
        document.dispatchEvent(event);
    }

    // Simulate real-time updates (in production, use WebSocket or SSE)
    simulateActivity() {
        const types = ['thinking', 'tool_creation', 'research', 'memory_update', 'file_change'];
        const type = types[Math.floor(Math.random() * types.length)];
        
        const activity = {
            id: Date.now().toString(),
            type: type,
            status: 'in_progress',
            title: this.generateTitle(type),
            details: 'Processing...',
            timestamp: new Date().toISOString(),
            data: {}
        };

        const id = this.addActivity(activity);

        // Simulate completion
        setTimeout(() => {
            this.updateActivity(id, {
                status: 'completed',
                details: 'Completed successfully'
            });
        }, Math.random() * 3000 + 1000);
    }

    generateTitle(type) {
        const titles = {
            'thinking': 'Analyzing user intent',
            'tool_creation': 'Creating web scraper tool',
            'research': 'Researching Python best practices',
            'memory_update': 'Storing interaction to episodic memory',
            'file_change': 'Modifying scraper.py'
        };
        return titles[type] || 'Processing...';
    }
}
