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
        
        // Initialize Gauntlet Dashboard
        if (typeof GauntletDashboard !== 'undefined') {
            window.gauntletDashboard = new GauntletDashboard(this);
        }
        
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
            'apps-repos': document.getElementById('view-apps-repos'),
            repos: document.getElementById('view-repos'),
            tools: document.getElementById('view-tools'),
            goals: document.getElementById('view-goals'),
            benchmarks: document.getElementById('view-benchmarks')
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
        
        // Add repo button (legacy)
        const addRepoBtnLegacy = document.getElementById('add-repo-btn-legacy');
        if (addRepoBtnLegacy) {
            addRepoBtnLegacy.addEventListener('click', () => this.showRepoConnectDialog());
        }
        
        // Create app button
        const createAppBtn = document.getElementById('create-app-btn');
        if (createAppBtn) {
            createAppBtn.addEventListener('click', () => this.showCreateAppDialog());
        }
        
        // Apps & Repos tab switching
        const appsReposTabs = document.querySelectorAll('.apps-repos-tab');
        appsReposTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const subtab = tab.getAttribute('data-subtab');
                this.switchAppsReposTab(subtab);
            });
        });
        
        // Project type filters
        const filterBtns = document.querySelectorAll('.project-filter-btn');
        filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const filter = btn.getAttribute('data-filter');
                this.filterProjects(filter);
                
                // Update active state
                filterBtns.forEach(b => {
                    b.style.background = '#161b22';
                    b.style.border = '1px solid #30363d';
                    b.style.color = '#8b949e';
                });
                btn.style.background = '#30363d';
                btn.style.border = 'none';
                btn.style.color = '#e6edf3';
            });
        });
        
        // Create goal button
        const createGoalBtn = document.getElementById('create-goal-btn');
        if (createGoalBtn) {
            createGoalBtn.addEventListener('click', () => this.showCreateGoalDialog());
        }
        
        // Activate Aether button
        const activateAetherBtn = document.getElementById('activate-aether-btn');
        if (activateAetherBtn) {
            activateAetherBtn.addEventListener('click', () => {
                if (window.gauntletDashboard) window.gauntletDashboard.activateAetherGauntlet();
            });
        }

        // Test Big Three button
        const testBigThreeBtn = document.getElementById('test-big-three-btn');
        if (testBigThreeBtn) {
            testBigThreeBtn.addEventListener('click', () => {
                if (window.gauntletDashboard) window.gauntletDashboard.testBigThreeTheory();
            });
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
    
    /**
     * Filter projects by type (all, app, tool, mcp, api)
     */
    filterProjects(filter) {
        const projectItems = document.querySelectorAll('.project-item');
        projectItems.forEach(item => {
            const type = item.getAttribute('data-project-type');
            if (filter === 'all' || type === filter) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
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
            if (viewName === 'apps-repos') {
                this.loadUserApps();
                this.loadUserRepos();
            } else if (viewName === 'repos') {
                this.loadUserRepos();
            } else if (viewName === 'goals') {
                this.loadUserGoals();
            } else if (viewName === 'benchmarks') {
                if (window.gauntletDashboard) window.gauntletDashboard.loadBenchmarks();
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
            'apps-repos': { icon: 'fa-rocket', title: 'Projects' },
            repos: { icon: 'fa-github', title: 'Repositories' },
            tools: { icon: 'fa-tools', title: 'Tools' },
            goals: { icon: 'fa-bullseye', title: 'Autonomous Goals' },
            benchmarks: { icon: 'fa-chart-line', title: 'Gauntlet Benchmarks' }
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
        const repoList = document.getElementById('repo-list') || document.getElementById('repo-list-legacy');
        if (!repoList) return;
        
        try {
            const apiKey = localStorage.getItem('aethermind_api_key') || localStorage.getItem('aether_api_key');
            if (!apiKey) {
                console.warn('No API key found for loading repos');
                repoList.innerHTML = '<li class="repo-item empty-state"><i class="fas fa-key"></i>Please log in first</li>';
                return;
            }
            const response = await fetch(this.backendUrl + '/api/user/repos', {
                headers: {
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
    
    switchAppsReposTab(subtab) {
        // Update tab buttons
        const tabs = document.querySelectorAll('.apps-repos-tab');
        tabs.forEach(tab => {
            if (tab.getAttribute('data-subtab') === subtab) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });
        
        // Update content
        const contents = document.querySelectorAll('.apps-repos-content');
        contents.forEach(content => {
            if (content.id === 'subtab-' + subtab) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });
    }
    
    async loadUserApps() {
        // Now loads ALL project types (apps, tools, mcp, api)
        await this.loadUserProjects();
    }
    
    async loadUserProjects() {
        const projectsList = document.getElementById('my-apps-list');
        if (!projectsList) return;
        
        try {
            const apiKey = localStorage.getItem('aethermind_api_key') || localStorage.getItem('aether_api_key');
            if (!apiKey) {
                projectsList.innerHTML = this.renderEmptyProjects('Please log in first', 'fa-key');
                return;
            }
            
            // Try new unified endpoint first
            let response = await fetch(this.backendUrl + '/v1/projects/list', {
                headers: { 'X-Aether-Key': apiKey }
            });
            
            // Fallback to legacy endpoint
            if (!response.ok) {
                response = await fetch(this.backendUrl + '/v1/apps/list', {
                    headers: { 'X-Aether-Key': apiKey }
                });
            }
            
            if (response.ok) {
                const data = await response.json();
                const projects = data.projects || data.apps || [];
                
                if (projects.length > 0) {
                    projectsList.innerHTML = projects.map(p => this.renderProjectItem(p)).join('');
                } else {
                    projectsList.innerHTML = this.renderEmptyProjects('No projects yet', 'fa-rocket');
                }
            } else {
                projectsList.innerHTML = this.renderEmptyProjects('No projects yet', 'fa-rocket');
            }
        } catch (error) {
            console.error('Error loading projects:', error);
            projectsList.innerHTML = this.renderEmptyProjects('No projects yet', 'fa-rocket');
        }
    }
    
    renderEmptyProjects(message, icon) {
        return `
            <div class="app-item empty-state" style="border: 1px dashed var(--border-color); cursor: default;">
                <i class="fas ${icon}" style="color: var(--text-muted);"></i>
                <div class="app-item-info">
                    <div class="app-item-desc">${message}</div>
                    <div class="app-item-hint">Create apps, tools, MCP servers, or APIs!</div>
                </div>
            </div>`;
    }
    
    renderProjectItem(project) {
        const typeConfig = {
            app: { icon: 'fa-globe', color: '#8b5cf6', label: 'APP' },
            tool: { icon: 'fa-wrench', color: '#f59e0b', label: 'TOOL' },
            mcp: { icon: 'fa-server', color: '#10b981', label: 'MCP' },
            api: { icon: 'fa-plug', color: '#3b82f6', label: 'API' }
        };
        
        const type = project.project_type || 'app';
        const config = typeConfig[type] || typeConfig.app;
        
        const statusColors = {
            running: '#10b981',
            stopped: '#6b7280',
            created: '#3b82f6',
            building: '#f59e0b',
            error: '#ef4444'
        };
        const status = project.status || 'created';
        const statusColor = statusColors[status] || '#6b7280';
        
        return `
            <div class="app-item project-item" data-project-type="${type}" onclick="window.dynamicSidebar.openProject('${project.id}', '${type}')">
                <i class="fas ${config.icon}" style="color: ${config.color};"></i>
                <div class="app-item-info">
                    <div class="app-item-name">${project.name}</div>
                    <div class="app-item-desc">${project.description || 'No description'}</div>
                </div>
                <div class="app-item-status" style="background: ${statusColor}20; color: ${statusColor}; font-size: 10px;">
                    ${config.label}
                </div>
            </div>`;
    }
    
    openProject(projectId, projectType) {
        console.log('Opening project:', projectId, 'type:', projectType);
        const config = window.PROJECT_TYPES?.[projectType] || {};
        
        if (window.sandboxManager) {
            window.sandboxManager.activate({
                projectId: projectId,
                projectType: projectType,
                previewMode: config.previewMode || 'iframe'
            });
        }
    }
    
    // Legacy method for backwards compatibility
    renderEmptyApps(message, icon) {
        return this.renderEmptyProjects(message, icon);
    }
    
    renderAppItem(app) {
        return this.renderProjectItem({ ...app, project_type: 'app' });
    }
    
    openApp(appId) {
        this.openProject(appId, 'app');
    }
    
    showCreateAppDialog() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);display:flex;justify-content:center;align-items:center;z-index:10000;';
        
        modal.innerHTML = `
        <div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:2rem;max-width:600px;width:90%;box-shadow:0 20px 60px rgba(0,0,0,0.7);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.5rem;">
                <h3 style="font-size:1.3rem;font-weight:700;color:#e6edf3;margin:0;">Create New Project</h3>
                <button onclick="this.closest('.modal-overlay').remove()" style="background:none;border:none;font-size:1.5rem;color:#8b949e;cursor:pointer;padding:0.25rem 0.5rem;"><i class="fas fa-times"></i></button>
            </div>
            
            <!-- Project Type Selection -->
            <label style="display:block;font-size:0.9rem;font-weight:600;color:#e6edf3;margin-bottom:0.5rem;">Project Type</label>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;margin-bottom:1.5rem;" id="project-type-grid">
                <button onclick="window.selectProjectType('app')" class="project-type-btn selected" data-type="app" style="display:flex;flex-direction:column;align-items:center;gap:0.5rem;padding:1rem;border:2px solid #8b5cf6;background:#8b5cf620;border-radius:8px;cursor:pointer;">
                    <i class="fas fa-globe" style="font-size:1.5rem;color:#8b5cf6;"></i>
                    <span style="font-size:0.8rem;color:#e6edf3;font-weight:600;">Web App</span>
                </button>
                <button onclick="window.selectProjectType('tool')" class="project-type-btn" data-type="tool" style="display:flex;flex-direction:column;align-items:center;gap:0.5rem;padding:1rem;border:1px solid #30363d;background:#161b22;border-radius:8px;cursor:pointer;">
                    <i class="fas fa-wrench" style="font-size:1.5rem;color:#f59e0b;"></i>
                    <span style="font-size:0.8rem;color:#8b949e;">Tool/Script</span>
                </button>
                <button onclick="window.selectProjectType('mcp')" class="project-type-btn" data-type="mcp" style="display:flex;flex-direction:column;align-items:center;gap:0.5rem;padding:1rem;border:1px solid #30363d;background:#161b22;border-radius:8px;cursor:pointer;">
                    <i class="fas fa-server" style="font-size:1.5rem;color:#10b981;"></i>
                    <span style="font-size:0.8rem;color:#8b949e;">MCP Server</span>
                </button>
                <button onclick="window.selectProjectType('api')" class="project-type-btn" data-type="api" style="display:flex;flex-direction:column;align-items:center;gap:0.5rem;padding:1rem;border:1px solid #30363d;background:#161b22;border-radius:8px;cursor:pointer;">
                    <i class="fas fa-plug" style="font-size:1.5rem;color:#3b82f6;"></i>
                    <span style="font-size:0.8rem;color:#8b949e;">API/Backend</span>
                </button>
            </div>
            
            <!-- Project Name -->
            <label style="display:block;font-size:0.9rem;font-weight:600;color:#e6edf3;margin-bottom:0.5rem;">Project Name</label>
            <input id="project-name" placeholder="my-awesome-project" style="width:100%;padding:0.75rem;background:#161b22;border:1px solid #30363d;border-radius:8px;color:#e6edf3;font-size:0.95rem;margin-bottom:1rem;box-sizing:border-box;" />
            
            <!-- Template Selection (changes based on project type) -->
            <label style="display:block;font-size:0.9rem;font-weight:600;color:#e6edf3;margin-bottom:0.5rem;">Template</label>
            <div id="template-grid" style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;margin-bottom:1rem;">
                <!-- Templates populated dynamically -->
            </div>
            
            <!-- Description -->
            <p style="font-size:0.85rem;color:#8b949e;margin-top:1rem;margin-bottom:0.5rem;">Describe what you want to build:</p>
            <textarea id="project-description" placeholder="E.g., A REST API that manages user authentication with JWT tokens..." style="width:100%;padding:0.75rem;background:#161b22;border:1px solid #30363d;border-radius:8px;color:#e6edf3;font-size:0.95rem;min-height:80px;resize:vertical;font-family:inherit;box-sizing:border-box;"></textarea>
            
            <button onclick="window.startBuildingProject()" style="margin-top:1.5rem;width:100%;padding:0.875rem;background:linear-gradient(135deg,#8b5cf6,#6366f1);border:none;border-radius:8px;color:white;font-weight:600;font-size:0.95rem;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:0.5rem;">
                <i class="fas fa-rocket"></i> Start Building
            </button>
        </div>`;
        
        document.body.appendChild(modal);
        
        // Initialize with app templates
        window.selectProjectType('app');
    }
    
    async loadUserGoals() {
        const goalsList = document.getElementById('goals-list');
        if (!goalsList) return;
        
        try {
            const apiKey = localStorage.getItem('aethermind_api_key') || localStorage.getItem('aether_api_key');
            if (!apiKey) {
                console.warn('No API key found for loading goals');
                goalsList.innerHTML = '<div class="tool-item empty-state" style="border: 1px dashed var(--border-color); cursor: default;"><i class="fas fa-key"></i><div class="tool-item-info"><div class="tool-item-desc">Please log in first</div></div></div>';
                return;
            }
            const response = await fetch(this.backendUrl + '/v1/goals/list', {
                headers: {
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

// =============================================
// SCALABLE PROJECT CREATION SYSTEM
// Supports: Apps, Tools, MCP Servers, APIs, etc.
// =============================================

// Project type configurations with templates
window.PROJECT_TYPES = {
    app: {
        name: 'Web App',
        icon: 'fa-globe',
        color: '#8b5cf6',
        description: 'Full-stack web applications',
        templates: [
            { id: 'blank', name: 'Blank', icon: 'fa-file', color: '#8b5cf6' },
            { id: 'react', name: 'React', icon: 'fab fa-react', color: '#61dafb' },
            { id: 'nextjs', name: 'Next.js', icon: 'fa-bolt', color: '#000000' },
            { id: 'flask', name: 'Flask', icon: 'fab fa-python', color: '#3776ab' }
        ],
        previewMode: 'iframe',
        buildCommand: 'npm run dev',
        deployTarget: 'vercel'
    },
    tool: {
        name: 'Tool/Script',
        icon: 'fa-wrench',
        color: '#f59e0b',
        description: 'Python/JS utilities & automation scripts',
        templates: [
            { id: 'blank', name: 'Blank', icon: 'fa-file', color: '#f59e0b' },
            { id: 'cli', name: 'CLI Tool', icon: 'fa-terminal', color: '#22c55e' },
            { id: 'data', name: 'Data Script', icon: 'fa-database', color: '#3b82f6' },
            { id: 'automation', name: 'Automation', icon: 'fa-robot', color: '#ec4899' }
        ],
        previewMode: 'terminal',
        buildCommand: 'python main.py',
        deployTarget: 'toolforge'
    },
    mcp: {
        name: 'MCP Server',
        icon: 'fa-server',
        color: '#10b981',
        description: 'Model Context Protocol servers',
        templates: [
            { id: 'blank', name: 'Blank', icon: 'fa-file', color: '#10b981' },
            { id: 'stdio', name: 'Stdio', icon: 'fa-exchange-alt', color: '#8b5cf6' },
            { id: 'sse', name: 'SSE', icon: 'fa-broadcast-tower', color: '#f59e0b' },
            { id: 'resource', name: 'Resource', icon: 'fa-folder-open', color: '#3b82f6' }
        ],
        previewMode: 'logs',
        buildCommand: 'python -m mcp_server',
        deployTarget: 'docker'
    },
    api: {
        name: 'API/Backend',
        icon: 'fa-plug',
        color: '#3b82f6',
        description: 'REST/GraphQL API backends',
        templates: [
            { id: 'blank', name: 'Blank', icon: 'fa-file', color: '#3b82f6' },
            { id: 'fastapi', name: 'FastAPI', icon: 'fa-bolt', color: '#009688' },
            { id: 'express', name: 'Express', icon: 'fab fa-node-js', color: '#68a063' },
            { id: 'graphql', name: 'GraphQL', icon: 'fa-project-diagram', color: '#e10098' }
        ],
        previewMode: 'api-tester',
        buildCommand: 'uvicorn main:app --reload',
        deployTarget: 'render'
    }
};

// Current project state
window.currentProjectConfig = {
    type: 'app',
    template: 'blank',
    name: '',
    description: ''
};

// Select project type and update templates
window.selectProjectType = function(type) {
    const config = window.PROJECT_TYPES[type];
    if (!config) return;
    
    window.currentProjectConfig.type = type;
    window.currentProjectConfig.template = 'blank';
    
    // Update type button styles
    document.querySelectorAll('.project-type-btn').forEach(btn => {
        const btnType = btn.dataset.type;
        if (btnType === type) {
            btn.style.border = `2px solid ${config.color}`;
            btn.style.background = `${config.color}20`;
            btn.querySelector('i').style.color = config.color;
            btn.querySelector('span').style.color = '#e6edf3';
        } else {
            btn.style.border = '1px solid #30363d';
            btn.style.background = '#161b22';
            btn.querySelector('i').style.color = window.PROJECT_TYPES[btnType]?.color || '#8b949e';
            btn.querySelector('span').style.color = '#8b949e';
        }
    });
    
    // Render templates for this type
    const templateGrid = document.getElementById('template-grid');
    if (templateGrid && config.templates) {
        templateGrid.innerHTML = config.templates.map((t, i) => `
            <button onclick="window.selectTemplate('${t.id}')" class="template-btn ${i === 0 ? 'selected' : ''}" data-template="${t.id}" 
                style="display:flex;flex-direction:column;align-items:center;gap:0.5rem;padding:1rem;border:${i === 0 ? '2px solid ' + config.color : '1px solid #30363d'};background:${i === 0 ? config.color + '20' : '#161b22'};border-radius:8px;cursor:pointer;">
                <i class="${t.icon.includes('fab') ? t.icon : 'fas ' + t.icon}" style="font-size:1.5rem;color:${t.color};"></i>
                <span style="font-size:0.75rem;color:${i === 0 ? '#e6edf3' : '#8b949e'};">${t.name}</span>
            </button>
        `).join('');
    }
    
    // Update placeholder text based on type
    const descInput = document.getElementById('project-description');
    if (descInput) {
        const placeholders = {
            app: 'E.g., A todo app with user authentication and real-time updates...',
            tool: 'E.g., A CLI tool that converts CSV files to JSON with filtering options...',
            mcp: 'E.g., An MCP server that provides access to my Notion workspace...',
            api: 'E.g., A REST API for managing a blog with posts, comments, and users...'
        };
        descInput.placeholder = placeholders[type] || 'Describe what you want to build...';
    }
};

// Select template within current type
window.selectTemplate = function(templateId) {
    const config = window.PROJECT_TYPES[window.currentProjectConfig.type];
    window.currentProjectConfig.template = templateId;
    
    document.querySelectorAll('.template-btn').forEach(btn => {
        const isSelected = btn.dataset.template === templateId;
        btn.style.border = isSelected ? `2px solid ${config.color}` : '1px solid #30363d';
        btn.style.background = isSelected ? `${config.color}20` : '#161b22';
        btn.querySelector('span').style.color = isSelected ? '#e6edf3' : '#8b949e';
    });
};

// Start building the project
window.startBuildingProject = function() {
    try {
        const nameInput = document.getElementById('project-name');
        const descInput = document.getElementById('project-description');
        
        const projectName = (nameInput && nameInput.value.trim()) ? nameInput.value.trim() : `my-${window.currentProjectConfig.type}-${Date.now()}`;
        const description = descInput ? descInput.value.trim() : '';
        
        const config = window.PROJECT_TYPES[window.currentProjectConfig.type];
        
        console.log('🚀 [StartProject] Building:', {
            type: window.currentProjectConfig.type,
            template: window.currentProjectConfig.template,
            name: projectName,
            description: description,
            previewMode: config.previewMode
        });
        
        // Close modal
        const modal = document.querySelector('.modal-overlay');
        if (modal) modal.remove();
        
        // Activate sandbox with full config
        setTimeout(() => {
            if (window.sandboxManager) {
                window.sandboxManager.activate({
                    projectType: window.currentProjectConfig.type,
                    template: window.currentProjectConfig.template,
                    name: projectName,
                    description: description,
                    previewMode: config.previewMode,
                    buildCommand: config.buildCommand,
                    deployTarget: config.deployTarget,
                    typeConfig: config
                });
                
                // Auto-send initial prompt if description provided
                if (description) {
                    setTimeout(() => {
                        const chatInput = document.getElementById('sandbox-chat-input');
                        if (chatInput) {
                            const prompt = buildInitialPrompt(window.currentProjectConfig.type, projectName, description, window.currentProjectConfig.template);
                            chatInput.value = prompt;
                            
                            // Trigger send
                            const sendBtn = document.getElementById('sandbox-send-btn');
                            if (sendBtn) sendBtn.click();
                        }
                    }, 500);
                }
            } else {
                console.error('❌ SandboxManager not initialized');
                alert('Sandbox not ready. Please refresh and try again.');
            }
        }, 100);
        
    } catch (error) {
        console.error('❌ Error starting project:', error);
        alert('Error: ' + error.message);
    }
};

// Build initial prompt based on project type
function buildInitialPrompt(type, name, description, template) {
    const prompts = {
        app: `Build a web app called "${name}": ${description}`,
        tool: `Create a tool/script called "${name}": ${description}. Make it a proper CLI tool with argument parsing.`,
        mcp: `Build an MCP server called "${name}": ${description}. Follow the Model Context Protocol specification.`,
        api: `Create a REST API called "${name}": ${description}. Include proper endpoints, error handling, and documentation.`
    };
    
    let prompt = prompts[type] || `Build a project called "${name}": ${description}`;
    
    if (template && template !== 'blank') {
        prompt += ` Use the ${template} framework/template.`;
    }
    
    return prompt;
}

// Legacy support for old createAppWithTemplate
window.createAppWithTemplate = function(template) {
    window.currentProjectConfig = { type: 'app', template: template };
    window.selectTemplate(template);
    
    const modal = document.querySelector('.modal-overlay');
    if (modal) modal.remove();
    
    setTimeout(() => {
        if (window.sandboxManager) {
            window.sandboxManager.activate({
                projectType: 'app',
                template: template,
                name: 'my-app-' + Date.now(),
                previewMode: 'iframe'
            });
        }
    }, 100);
};

// Legacy support
window.startBuildingApp = window.startBuildingProject;

// Initialize sidebar when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        window.dynamicSidebar = new DynamicSidebar();
    });
} else {
    window.dynamicSidebar = new DynamicSidebar();
}
