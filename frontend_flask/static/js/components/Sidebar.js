/**
 * Dynamic Sidebar Component
 * Handles navigation between repos, tools, and goals views
 */

class Sidebar {
    constructor() {
        this.sidebar = document.getElementById('sidebar');
        this.overlay = document.getElementById('sidebar-overlay');
        this.toggleBtn = document.getElementById('sidebar-toggle');
        this.closeBtn = document.getElementById('sidebar-close-btn');
        this.backBtn = document.getElementById('sidebar-back-btn');
        this.sidebarTitle = document.getElementById('sidebar-title');
        this.sidebarIcon = document.getElementById('sidebar-icon');
        
        this.currentView = 'main';
        this.viewHistory = [];
        
        this.init();
    }
    
    init() {
        // Toggle sidebar open/close
        this.toggleBtn.addEventListener('click', () => this.open());
        this.closeBtn.addEventListener('click', () => this.close());
        this.overlay.addEventListener('click', () => this.close());
        
        // Back button navigation
        this.backBtn.addEventListener('click', () => this.goBack());
        
        // Main menu navigation
        document.querySelectorAll('[data-view]').forEach(item => {
            item.addEventListener('click', () => {
                const view = item.getAttribute('data-view');
                this.navigateTo(view);
            });
        });
        
        // Add repo button
        document.getElementById('add-repo-btn')?.addEventListener('click', () => {
            this.showRepoConnectDialog();
        });
        
        // Create goal button
        document.getElementById('create-goal-btn')?.addEventListener('click', () => {
            this.showCreateGoalDialog();
        });
        
        // Tools search
        document.getElementById('tools-search')?.addEventListener('input', (e) => {
            this.filterTools(e.target.value);
        });
        
        // Goals search
        document.getElementById('goals-search')?.addEventListener('input', (e) => {
            this.filterGoals(e.target.value);
        });
        
        // Load initial data
        this.loadCoreTools();
        this.loadUserRepos();
        this.loadUserGoals();
    }
    
    open() {
        this.sidebar.classList.add('open');
        this.overlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
    
    close() {
        this.sidebar.classList.remove('open');
        this.overlay.classList.remove('active');
        document.body.style.overflow = '';
    }
    
    navigateTo(view) {
        // Save current view to history
        if (this.currentView !== view) {
            this.viewHistory.push(this.currentView);
        }
        
        // Hide all views
        document.querySelectorAll('.sidebar-view').forEach(v => {
            v.classList.remove('active');
        });
        
        // Show target view
        const targetView = document.getElementById(`view-${view}`);
        if (targetView) {
            targetView.classList.add('active');
            this.currentView = view;
            this.updateHeader(view);
        }
    }
    
    goBack() {
        if (this.viewHistory.length > 0) {
            const previousView = this.viewHistory.pop();
            
            // Hide current view
            document.querySelectorAll('.sidebar-view').forEach(v => {
                v.classList.remove('active');
            });
            
            // Show previous view
            const targetView = document.getElementById(`view-${previousView}`);
            if (targetView) {
                targetView.classList.add('active');
                this.currentView = previousView;
                this.updateHeader(previousView);
            }
        }
    }
    
    updateHeader(view) {
        const headers = {
            'main': { icon: 'fa-cube', title: 'Menu' },
            'repos': { icon: 'fab fa-github', title: 'Repositories' },
            'tools': { icon: 'fa-tools', title: 'Tools' },
            'goals': { icon: 'fa-bullseye', title: 'Goals' }
        };
        
        const header = headers[view] || headers.main;
        this.sidebarIcon.className = `fas ${header.icon}`;
        this.sidebarTitle.textContent = header.title;
        
        // Show/hide back button
        if (view === 'main') {
            this.backBtn.classList.remove('visible');
        } else {
            this.backBtn.classList.add('visible');
        }
    }
    
    async loadCoreTools() {
        const coreTools = [
            {
                name: 'Code Executor',
                description: 'Run Python/Bash code',
                icon: 'fa-play',
                type: 'mcp',
                enabled: true
            },
            {
                name: 'File Manager',
                description: 'Create and edit files',
                icon: 'fa-file-code',
                type: 'mcp',
                enabled: true
            },
            {
                name: 'Web Search',
                description: 'Search the internet',
                icon: 'fa-search',
                type: 'mcp',
                enabled: true
            },
            {
                name: 'Docker Manager',
                description: 'Manage containers',
                icon: 'fab fa-docker',
                type: 'mcp',
                enabled: true
            },
            {
                name: 'PyPI Search',
                description: 'Find Python packages',
                icon: 'fab fa-python',
                type: 'pypi',
                enabled: true
            }
        ];
        
        const container = document.getElementById('core-tools-list');
        container.innerHTML = coreTools.map(tool => this.renderToolItem(tool)).join('');
    }
    
    renderToolItem(tool) {
        return `
            <div class="tool-item" data-tool="${tool.name}">
                <div class="tool-item-icon">
                    <i class="fas ${tool.icon}"></i>
                </div>
                <div class="tool-item-info">
                    <div class="tool-item-name">${tool.name}</div>
                    <div class="tool-item-desc">${tool.description}</div>
                </div>
                ${tool.enabled ? '<span class="tool-item-badge">Active</span>' : ''}
            </div>
        `;
    }
    
    async loadUserRepos() {
        try {
            const response = await fetch('/api/user/repos');
            if (response.ok) {
                const repos = await response.json();
                this.renderRepos(repos);
            }
        } catch (error) {
            console.log('No repos loaded yet:', error);
        }
    }
    
    renderRepos(repos) {
        const container = document.getElementById('repo-list');
        
        if (!repos || repos.length === 0) {
            container.innerHTML = `
                <li class="repo-item empty-state">
                    <i class="fas fa-inbox"></i>
                    No repositories connected
                </li>
            `;
            return;
        }
        
        container.innerHTML = repos.map(repo => `
            <li class="repo-item" data-repo="${repo.full_name}">
                <i class="fab fa-github"></i>
                <div class="repo-item-info">
                    <div class="repo-item-name">${repo.name}</div>
                    <div class="repo-item-desc">${repo.description || 'No description'}</div>
                </div>
            </li>
        `).join('');
    }
    
    async loadUserGoals() {
        try {
            const apiKey = localStorage.getItem('aether_api_key');
            if (!apiKey) return;
            
            const response = await fetch('/api/v1/goals/list', {
                headers: {
                    'Authorization': `Bearer ${apiKey}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.renderGoals(data.goals || []);
            }
        } catch (error) {
            console.log('Could not load goals:', error);
        }
    }
    
    renderGoals(goals) {
        const container = document.getElementById('goals-list');
        
        if (!goals || goals.length === 0) {
            container.innerHTML = `
                <div class="tool-item empty-state" style="border: 1px dashed var(--border-color); cursor: default;">
                    <i class="fas fa-bullseye" style="color: var(--text-muted);"></i>
                    <div class="tool-item-info">
                        <div class="tool-item-desc">No goals created yet</div>
                    </div>
                </div>
            `;
            return;
        }
        
        container.innerHTML = goals.map(goal => {
            const statusColors = {
                'pending': '#f59e0b',
                'in_progress': '#3b82f6',
                'completed': '#10b981',
                'failed': '#ef4444'
            };
            
            const statusIcons = {
                'pending': 'fa-clock',
                'in_progress': 'fa-spinner fa-spin',
                'completed': 'fa-check-circle',
                'failed': 'fa-exclamation-circle'
            };
            
            return `
                <div class="tool-item" data-goal="${goal.goal_id}" onclick="window.viewGoalDetails('${goal.goal_id}')">
                    <div class="tool-item-icon" style="background: ${statusColors[goal.status] || '#6b7280'};">
                        <i class="fas ${statusIcons[goal.status] || 'fa-circle'}"></i>
                    </div>
                    <div class="tool-item-info">
                        <div class="tool-item-name">${goal.description.substring(0, 40)}${goal.description.length > 40 ? '...' : ''}</div>
                        <div class="tool-item-desc">
                            ${goal.progress ? `${goal.progress.percentage.toFixed(0)}% complete` : goal.status}
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }
    
    filterTools(query) {
        const tools = document.querySelectorAll('#view-tools .tool-item:not(.empty-state)');
        const searchLower = query.toLowerCase();
        
        tools.forEach(tool => {
            const name = tool.querySelector('.tool-item-name')?.textContent.toLowerCase() || '';
            const desc = tool.querySelector('.tool-item-desc')?.textContent.toLowerCase() || '';
            
            if (name.includes(searchLower) || desc.includes(searchLower)) {
                tool.style.display = '';
            } else {
                tool.style.display = 'none';
            }
        });
    }
    
    filterGoals(query) {
        const goals = document.querySelectorAll('#goals-list .tool-item:not(.empty-state)');
        const searchLower = query.toLowerCase();
        
        goals.forEach(goal => {
            const name = goal.querySelector('.tool-item-name')?.textContent.toLowerCase() || '';
            
            if (name.includes(searchLower)) {
                goal.style.display = '';
            } else {
                goal.style.display = 'none';
            }
        });
    }
    
    showRepoConnectDialog() {
        // Create modal for GitHub OAuth
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 500px;">
                <div class="modal-header">
                    <h3><i class="fab fa-github"></i> Connect Repository</h3>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <p style="color: var(--text-muted); margin-bottom: 1.5rem;">
                        Connect a GitHub repository to give AetherMind access to your code.
                    </p>
                    <div style="display: flex; flex-direction: column; gap: 1rem;">
                        <button class="sidebar-add-btn" onclick="window.connectGitHubRepo('public')">
                            <i class="fas fa-globe"></i>
                            Connect Public Repository
                        </button>
                        <button class="sidebar-add-btn" style="background: #6b7280;" onclick="window.connectGitHubRepo('private')">
                            <i class="fas fa-lock"></i>
                            Connect Private Repository
                        </button>
                    </div>
                    <p style="color: var(--text-muted); font-size: 0.8rem; margin-top: 1rem;">
                        <i class="fas fa-info-circle"></i> You'll be redirected to GitHub to authorize access.
                    </p>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    showCreateGoalDialog() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 500px;">
                <div class="modal-header">
                    <h3><i class="fas fa-bullseye"></i> Create Autonomous Goal</h3>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <label style="display: block; margin-bottom: 0.5rem; font-weight: 600;">Goal Description</label>
                    <textarea 
                        id="goal-description" 
                        style="width: 100%; padding: 0.75rem; background: rgba(255,255,255,0.05); border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-primary); min-height: 100px; font-family: inherit; resize: vertical;"
                        placeholder="E.g., Create a Flask todo app with SQLite database"
                    ></textarea>
                    
                    <label style="display: block; margin: 1rem 0 0.5rem; font-weight: 600;">Priority</label>
                    <select 
                        id="goal-priority"
                        style="width: 100%; padding: 0.75rem; background: rgba(255,255,255,0.05); border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-primary);"
                    >
                        <option value="5">5 - Normal</option>
                        <option value="8">8 - High</option>
                        <option value="3">3 - Low</option>
                        <option value="10">10 - Critical</option>
                    </select>
                    
                    <button class="sidebar-add-btn" style="margin-top: 1.5rem;" onclick="window.submitGoal()">
                        <i class="fas fa-rocket"></i>
                        Create Goal
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
}

// Global functions for modal actions
window.connectGitHubRepo = function(type) {
    console.log(`Connecting ${type} repository...`);
    // TODO: Implement GitHub OAuth flow
    window.location.href = `/auth/github?type=${type}`;
};

window.submitGoal = async function() {
    const description = document.getElementById('goal-description').value;
    const priority = parseInt(document.getElementById('goal-priority').value);
    
    if (!description.trim()) {
        alert('Please enter a goal description');
        return;
    }
    
    try {
        const apiKey = localStorage.getItem('aether_api_key');
        const response = await fetch('/api/v1/goals/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                description: description,
                priority: priority,
                metadata: { domain: 'general' }
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            alert(`Goal created! ID: ${result.goal_id}\n\nAetherMind will work on this autonomously in the background.`);
            document.querySelector('.modal-overlay').remove();
            
            // Reload goals
            window.sidebar.loadUserGoals();
        } else {
            alert('Failed to create goal. Please try again.');
        }
    } catch (error) {
        console.error('Error creating goal:', error);
        alert('Error creating goal. Please check console.');
    }
};

window.viewGoalDetails = function(goalId) {
    console.log('Viewing goal:', goalId);
    // TODO: Implement goal details view
    alert(`Goal details for ${goalId}\n\n(Coming soon: detailed subtask view)`);
};

// Initialize sidebar when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.sidebar = new Sidebar();
    });
} else {
    window.sidebar = new Sidebar();
}
