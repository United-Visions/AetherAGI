// NotificationManager.js - Centralized notification system
// Handles both push notifications and in-app notifications

export class NotificationManager {
    constructor() {
        this.notifications = [];
        this.unreadCount = 0;
        this.listeners = [];
        
        // Load from storage
        this.load();
        
        // Request browser notification permission
        this.requestPermission();
    }

    load() {
        try {
            const stored = localStorage.getItem('aethermind_notifications');
            if (stored) {
                this.notifications = JSON.parse(stored);
                this.unreadCount = this.notifications.filter(n => !n.read).length;
            }
        } catch (err) {
            console.error('Failed to load notifications:', err);
        }
    }

    save() {
        try {
            // Keep only last 50 notifications
            const toSave = this.notifications.slice(0, 50);
            localStorage.setItem('aethermind_notifications', JSON.stringify(toSave));
        } catch (err) {
            console.error('Failed to save notifications:', err);
        }
    }

    async requestPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            // Don't ask immediately, wait for user interaction
            this.permissionPending = true;
        }
    }

    async askPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            const permission = await Notification.requestPermission();
            return permission === 'granted';
        }
        return Notification.permission === 'granted';
    }

    add(notification) {
        const notif = {
            id: notification.id || `notif_${Date.now()}`,
            type: notification.type || 'info',
            title: notification.title || 'AetherMind',
            message: notification.message || '',
            timestamp: notification.timestamp || new Date().toISOString(),
            read: false,
            data: notification.data || {}
        };
        
        this.notifications.unshift(notif);
        this.unreadCount++;
        
        // Update UI
        this.updateBadge();
        this.renderList();
        
        // Save
        this.save();
        
        // Browser notification if permitted and app not focused
        if (document.hidden && Notification.permission === 'granted') {
            this.showBrowserNotification(notif);
        }
        
        // Notify listeners
        this.listeners.forEach(fn => fn(notif));
        
        return notif.id;
    }

    showBrowserNotification(notif) {
        const browserNotif = new Notification(notif.title, {
            body: notif.message,
            icon: '/static/img/aether-icon.png',
            badge: '/static/img/aether-badge.png',
            tag: notif.id,
            requireInteraction: notif.type === 'task_complete'
        });
        
        browserNotif.onclick = () => {
            window.focus();
            this.markRead(notif.id);
        };
    }

    markRead(id) {
        const notif = this.notifications.find(n => n.id === id);
        if (notif && !notif.read) {
            notif.read = true;
            this.unreadCount--;
            this.updateBadge();
            this.save();
        }
    }

    markAllRead() {
        this.notifications.forEach(n => n.read = true);
        this.unreadCount = 0;
        this.updateBadge();
        this.save();
    }

    remove(id) {
        const index = this.notifications.findIndex(n => n.id === id);
        if (index !== -1) {
            const notif = this.notifications[index];
            if (!notif.read) this.unreadCount--;
            this.notifications.splice(index, 1);
            this.updateBadge();
            this.renderList();
            this.save();
        }
    }

    clear() {
        this.notifications = [];
        this.unreadCount = 0;
        this.updateBadge();
        this.renderList();
        this.save();
    }

    updateBadge() {
        const badge = document.getElementById('notification-badge');
        if (badge) {
            badge.textContent = this.unreadCount;
            badge.classList.toggle('hidden', this.unreadCount === 0);
        }
    }

    renderList() {
        const container = document.getElementById('notification-list');
        if (!container) return;
        
        if (this.notifications.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-bell-slash"></i>
                    <p>No notifications yet</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = this.notifications.map(notif => `
            <div class="notification-item ${notif.read ? '' : 'unread'}" data-id="${notif.id}">
                <div class="notif-icon">
                    <i class="fas ${this.getIcon(notif.type)}"></i>
                </div>
                <div class="notif-content">
                    <h4>${this.escapeHtml(notif.title)}</h4>
                    <p>${this.escapeHtml(notif.message)}</p>
                    <time>${this.formatTime(notif.timestamp)}</time>
                </div>
                <button class="notif-dismiss" onclick="event.stopPropagation(); window.aether.notifications.remove('${notif.id}')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `).join('');
        
        // Add click handlers
        container.querySelectorAll('.notification-item').forEach(item => {
            item.addEventListener('click', () => {
                const id = item.dataset.id;
                this.markRead(id);
                item.classList.remove('unread');
                
                // If it's a task notification, show task details
                const notif = this.notifications.find(n => n.id === id);
                if (notif?.data?.taskId) {
                    window.aether?.showTaskDetails(notif.data.taskId);
                }
            });
        });
    }

    showTasks(tasks) {
        const container = document.getElementById('notification-list');
        if (!container) return;
        
        if (tasks.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-tasks"></i>
                    <p>No background tasks running</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = `
            <div class="tasks-header">
                <h4>Background Tasks</h4>
            </div>
            ${tasks.map(task => `
                <div class="notification-item task-item" data-task-id="${task.id}">
                    <div class="notif-icon ${task.status}">
                        <i class="fas ${this.getTaskIcon(task.status)}"></i>
                    </div>
                    <div class="notif-content">
                        <h4>${this.escapeHtml(task.name)}</h4>
                        <p>${task.status === 'running' ? 'In progress...' : task.status}</p>
                        ${task.progress ? `<div class="task-progress"><div class="progress-bar" style="width: ${task.progress}%"></div></div>` : ''}
                    </div>
                </div>
            `).join('')}
        `;
    }

    getIcon(type) {
        const icons = {
            'task_complete': 'fa-check-circle',
            'task_error': 'fa-exclamation-circle',
            'research': 'fa-search',
            'tool_created': 'fa-wrench',
            'goal_update': 'fa-bullseye',
            'memory': 'fa-brain',
            'info': 'fa-info-circle',
            'warning': 'fa-exclamation-triangle',
            'error': 'fa-times-circle'
        };
        return icons[type] || 'fa-bell';
    }

    getTaskIcon(status) {
        const icons = {
            'running': 'fa-spinner fa-spin',
            'completed': 'fa-check',
            'error': 'fa-exclamation',
            'pending': 'fa-clock'
        };
        return icons[status] || 'fa-circle';
    }

    formatTime(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;
        return date.toLocaleDateString();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    onNotification(callback) {
        this.listeners.push(callback);
    }
}
