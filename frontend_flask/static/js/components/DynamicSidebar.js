/**
 * DynamicSidebar.js
 * Collapsible sidebar with navigation for Repos, Tools, and Goals
 */

class DynamicSidebar {
    constructor() {
        this.currentView = 'main';
        this.viewHistory = [];
        this.isOpen = false;
        
        // Backend API URL - FastAPI runs on port 8000
        this.backendUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://127.0.0.1:8000'
            : 'https://aetheragi.onrender.com';
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadCoreTools();
        
        console.log('DynamicSidebar initialized with backend:', this.backendUrl);
    }
    
    initializeElements() {
        this.sidebar = document.getElementById('sidebar');
        this.overlay = document.getElementById('sidebar-overlay');
        this.toggleBtn = document.getElementById('sidebar-toggle');
        this.closeBtn = document.getElementById('sidebar-close-btn');
        this.backBtn = document.getElementById('sidebar-back-btn');
        this.headerIcon = document.getElementById('sidebar-icon');
        this.headerTitle = document.getElementById('sidebar-title');
        
        // Views
        this.views = {
            main: document.getElementById('view-main'),
            repos: document.getElementById('view-repos'),
            tools: document.getElementById('view-tools'),
            goals: document.getElementById('view-goals')
        };
    }
    
    attachEventListeners() {
        // Toggle sidebar
        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', () => this.toggle());
        }
        
        // Close sidebar
        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => this.close());
        }
        
        // Click overlay to close
        if (this.overlay) {
            this.overlay.addEventListener('click', () => this.close());
        }
        
        // Back button
        if (this.backBtn) {
            this.backBtn.addEventListener('click', () => this.goBack());
        }
        
        // Menu items navigation
        const menuItems = document.querySelectorAll('.sidebar-menu-item');
        menuItems.forEach(item => {
            item.addEventListener('click', () => {
                const view = item.getAttribute('data-view');
                if (view) this.navigateTo(view);
            });
        });
        
        // Add repo button
        const addRepoBtn = document.getElementById('add-repo-btn');
        if (addRepoBtn) {
            addRepoBtn.addEventListener('click', () => this.showRepoConnectDialog());
        }
        
        // Create goal button
        const createGoalBtn = document.getElementById('create-goal-btn');
        if (createGoalBtn) {
            createGoalBtn.addEventListener('click', () => this.showCreateGoalDialog());
        }
        
        // Search tools
        const toolsSearch = document.getElementById('tools-search');
        if (toolsSearch) {
            toolsSearch.addEventListener('input', (e) => this.filterTools(e.target.value));
        }
        
        // Search goals
        const goalsSearch = document.getElementById('goals-search');
        if (goalsSearch) {
            goalsSearch.addEventListener('input', (e) => this.filterGoals(e.target.value));
        }
    }
    
    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }
    
    open() {
        this.sidebar.classList.add('open');
        this.overlay.classList.add('active');
        this.isOpen = true;
    }
    
    close() {
        this.sidebar.classList.remove('open');
        this.overlay.classList.remove('active');
        this.isOpen = false;
    }
    
    navigateTo(viewName) {
        // Hide current view
        Object.values(this.views).forEach(view => {
            if (view) view.classList.remove('active');
        });
        
        // Show new view
        if (this.views[viewName]) {
            this.views[viewName].classList.add('active');
            this.viewHistory.push(this.currentView);
            this.currentView = viewName;
            this.updateHeader();
            
            // Load data for specific views
            if (viewName === 'repos') {
                this.loadUserRepos();
            } else if (viewName === 'goals') {
                this.loadUserGoals();
            }
        }
    }
    
    goBack() {
        if (this.viewHistory.length > 0) {
            const previousView = this.viewHistory.pop();
            
            // Hide current view
            Object.values(this.views).forEach(view => {
                if (view) view.classList.remove('active');
            });
            
            // Show previous view
            if (this.views[previousView]) {
                this.views[previousView].classList.add('active');
                this.currentView = previousView;
                this.updateHeader();
            }
        }
    }
    
    updateHeader() {
        const headers = {
            main: { icon: 'fa-cube', title: 'Menu' },
            repos: { icon: 'fa-github', title: 'Repositories' },
            tools: { icon: 'fa-tools', title: 'Tools' },
            goals: { icon: 'fa-bullseye', title: 'Autonomous Goals' }
        };
        
        const header = headers[this.currentView] || headers.main;
        this.headerIcon.className = 'fas ' + header.icon;
        this.headerTitle.textContent = header.title;
        
        // Show/hide back button
        if (this.currentView === 'main') {
            this.backBtn.style.display = 'none';
        } else {
            this.backBtn.style.display = 'flex';
        }
    }
    
    loadCoreTools() {
        const coreToolsList = document.getElementById('core-tools-list');
        if (!coreToolsList) return;
        
        const coreTools = [
            { name: 'Code Executor', desc: 'Run Python/Bash code', type: 'MCP', icon: 'fa-code' },
            { name: 'File Manager', desc: 'Create and edit files', type: 'MCP', icon: 'fa-folder' },
            { name: 'Web Search', desc: 'Search the internet', type: 'MCP', icon: 'fa-search' },
            { name: 'Docker Manager', desc: 'Manage containers', type: 'MCP', icon: 'fa-docker' },
            { name: 'Git Search', desc: 'Search repositories', type: 'MCP', icon: 'fa-git-alt' }
        ];
        
        coreToolsList.innerHTML = coreTools.map(tool => {
            return '<div class="tool-item">' +
                '<i class="fas ' + tool.icon + '" style="color: var(--accent-color);"></i>' +
                '<div class="tool-item-info">' +
                '<div class="tool-item-name">' + tool.name + '</div>' +
                '<div class="tool-item-desc">' + tool.desc + '</div>' +
                '</div>' +
                '<div class="tool-item-badge">' + tool.type + '</div>' +
                '</div>';
        }).join('');
    }
    
    async loadUserRepos() {
        const repoList = document.getElementById('repo-list');
        if (!repoList) return;
        
        try {
            const apiKey = localStorage.getItem('aether_api_key') || localStorage.getItem('aethermind_api_key');
            if (!apiKey) {
                console.warn('No API key found for loading repos');
                repoList.innerHTML = '<li class="repo-item empty-state"><i class="fas fa-key"></i>Please log in first</li>';
                return;
            }
            const response = await fetch(this.backendUrl + '/api/user/repos', {
                headers: {
                    'Authorization': 'Bearer ' + apiKey,
                    'X-Aether-Key': apiKey
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.repos && data.repos.length > 0) {
                    repoList.innerHTML = data.repos.map(repo => {
                        return '<li class="repo-item">' +
                            '<i class="fab fa-github"></i>' +
                            '<div class="repo-info">' +
                            '<div class="repo-name">' + repo.name + '</div>' +
                            '<div class="repo-desc">' + (repo.description || 'No description') + '</div>' +
                            '</div>' +
                            '</li>';
                    }).join('');
                } else {
                    repoList.innerHTML = '<li class="repo-item empty-state"><i class="fas fa-inbox"></i>No repositories connected</li>';
                }
            }
        } catch (error) {
            console.error('Error loading repos:', error);
            repoList.innerHTML = '<li class="repo-item empty-state"><i class="fas fa-exclamation-triangle"></i>Error loading repositories</li>';
        }
    }
    
    async loadUserGoals() {
        const goalsList = document.getElementById('goals-list');
        if (!goalsList) return;
        
        try {
            const apiKey = localStorage.getItem('aether_api_key') || localStorage.getItem('aethermind_api_key');
            if (!apiKey) {
                console.warn('No API key found for loading goals');
                goalsList.innerHTML = '<div class="tool-item empty-state" style="border: 1px dashed var(--border-color); cursor: default;"><i class="fas fa-key"></i><div class="tool-item-info"><div class="tool-item-desc">Please log in first</div></div></div>';
                return;
            }
            const response = await fetch(this.backendUrl + '/v1/goals/list', {
                headers: {
                    'Authorization': 'Bearer ' + apiKey,
                    'X-Aether-Key': apiKey
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.goals && data.goals.length > 0) {
                    goalsList.innerHTML = data.goals.map(goal => {
                        const statusColors = {
                            pending: '#fbbf24',
                            in_progress: '#3b82f6',
                            completed: '#10b981',
                            failed: '#ef4444'
                        };
                        const statusColor = statusColors[goal.status] || '#6b7280';
                        
                        return '<div class="tool-item" onclick="viewGoalDetails(\'' + goal.goal_id + '\')" style="cursor: pointer;">' +
                            '<i class="fas fa-bullseye" style="color: ' + statusColor + ';"></i>' +
                            '<div class="tool-item-info">' +
                            '<div class="tool-item-name">' + goal.description.substring(0, 50) + '...</div>' +
                            '<div class="tool-item-desc">Progress: ' + Math.round(goal.progress) + '%</div>' +
                            '</div>' +
                            '<div class="tool-item-badge" style="background: ' + statusColor + '20; border-color: ' + statusColor + '40; color: ' + statusColor + ';">' + 
                            goal.status.toUpperCase() + 
                            '</div>' +
                            '</div>';
                    }).join('');
                } else {
                    goalsList.innerHTML = '<div class="tool-item empty-state" style="border: 1px dashed var(--border-color); cursor: default;"><i class="fas fa-inbox"></i><div class="tool-item-info"><div class="tool-item-desc">No autonomous goals created yet</div></div></div>';
                }
            }
        } catch (error) {
            console.error('Error loading goals:', error);
            goalsList.innerHTML = '<div class="tool-item empty-state" style="border: 1px dashed var(--border-color); cursor: default;"><i class="fas fa-exclamation-triangle"></i><div class="tool-item-info"><div class="tool-item-desc">Error loading goals</div></div></div>';
        }
    }
    
    filterTools(query) {
        const toolItems = document.querySelectorAll('#core-tools-list .tool-item, #user-tools-list .tool-item');
        const lowerQuery = query.toLowerCase();
        
        toolItems.forEach(item => {
            const name = item.querySelector('.tool-item-name');
            const desc = item.querySelector('.tool-item-desc');
            
            if (name && desc) {
                const nameText = name.textContent.toLowerCase();
                const descText = desc.textContent.toLowerCase();
                
                if (nameText.includes(lowerQuery) || descText.includes(lowerQuery)) {
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            }
        });
    }
    
    filterGoals(query) {
        const goalItems = document.querySelectorAll('#goals-list .tool-item');
        const lowerQuery = query.toLowerCase();
        
        goalItems.forEach(item => {
            const name = item.querySelector('.tool-item-name');
            
            if (name) {
                const nameText = name.textContent.toLowerCase();
                
                if (nameText.includes(lowerQuery)) {
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            }
        });
    }
    
    showRepoConnectDialog() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = '<div class="modal-content">' +
            '<div class="modal-header">' +
            '<h3 class="modal-title">Connect Repository</h3>' +
            '<button class="modal-close" onclick="this.closest(\'.modal-overlay\').remove()"><i class="fas fa-times"></i></button>' +
            '</div>' +
            '<div class="modal-body">' +
            '<label class="modal-label">Repository Type</label>' +
            '<div style="display: flex; gap: 1rem; margin-top: 0.5rem;">' +
            '<button class="modal-btn modal-btn-primary" onclick="connectGitHubRepo(\'public\')"><i class="fab fa-github"></i> Public Repo</button>' +
            '<button class="modal-btn modal-btn-secondary" onclick="connectGitHubRepo(\'private\')"><i class="fas fa-lock"></i> Private Repo</button>' +
            '</div>' +
            '<p class="modal-hint" style="margin-top: 1rem;">Connect your GitHub repositories to let AetherMind analyze and work with your code.</p>' +
            '</div>' +
            '</div>';
        
        document.body.appendChild(modal);
    }
    
    showCreateGoalDialog() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = '<div class="modal-content">' +
            '<div class="modal-header">' +
            '<h3 class="modal-title">Create Autonomous Goal</h3>' +
            '<button class="modal-close" onclick="this.closest(\'.modal-overlay\').remove()"><i class="fas fa-times"></i></button>' +
            '</div>' +
            '<div class="modal-body">' +
            '<label class="modal-label">Goal Description</label>' +
            '<textarea id="goal-description" class="modal-textarea" placeholder="E.g., Create a Flask todo app with SQLite database"></textarea>' +
            '<label class="modal-label" style="margin-top: 1rem;">Priority</label>' +
            '<select id="goal-priority" class="modal-select">' +
            '<option value="5">5 - Normal</option>' +
            '<option value="8">8 - High</option>' +
            '<option value="3">3 - Low</option>' +
            '<option value="10">10 - Critical</option>' +
            '</select>' +
            '<button class="modal-btn modal-btn-primary" style="margin-top: 1.5rem; width: 100%;" onclick="submitGoal()"><i class="fas fa-rocket"></i> Create Goal</button>' +
            '</div>' +
            '</div>';
        
        document.body.appendChild(modal);
    }
}

// Global functions for modal actions
window.connectGitHubRepo = function(type) {
    console.log('Connecting ' + type + ' repository...');
    alert('GitHub OAuth integration coming soon!\n\nThis will redirect you to GitHub to authorize AetherMind.');
    document.querySelector('.modal-overlay').remove();
};

window.submitGoal = async function() {
    const description = document.getElementById('goal-description').value;
    const priority = parseInt(document.getElementById('goal-priority').value);
    
    if (!description.trim()) {
        alert('Please enter a goal description');
        return;
    }
    
    try {
        const apiKey = localStorage.getItem('aether_api_key') || localStorage.getItem('aethermind_api_key');
        const backendUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://127.0.0.1:8000'
            : 'https://aetheragi.onrender.com';
        const response = await fetch(backendUrl + '/v1/goals/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
                'X-Aether-Key': apiKey
            },
            body: JSON.stringify({
                description: description,
                priority: priority,
                metadata: { domain: 'general' }
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            alert('Goal created! ID: ' + result.goal_id + '\n\nAetherMind will work on this autonomously in the background.');
            document.querySelector('.modal-overlay').remove();
            
            // Reload goals if on goals view
            if (window.dynamicSidebar && window.dynamicSidebar.currentView === 'goals') {
                window.dynamicSidebar.loadUserGoals();
            }
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
    alert('Goal details for ' + goalId + '\n\n(Coming soon: detailed subtask view)');
};

// Initialize sidebar when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        window.dynamicSidebar = new DynamicSidebar();
    });
} else {
    window.dynamicSidebar = new DynamicSidebar();
}
