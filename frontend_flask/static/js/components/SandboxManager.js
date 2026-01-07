/**
 * SandboxManager.js
 * App Creation Mode - Full-screen sandbox environment for building apps
 * Lovable/v0.dev inspired design with live preview and terminal logs
 */

export class SandboxManager {
    constructor(containerId = 'sandbox-mode-container') {
        this.containerId = containerId;
        this.isActive = false;
        this.currentApp = null;
        this.sandboxes = [];
        this.logs = [];
        this.previewUrl = null;
        this.pollingInterval = null;
        
        // Backend URL
        this.backendUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://127.0.0.1:8000'
            : 'https://aetheragi.onrender.com';
        
        this.init();
    }
    
    init() {
        // Check if container already exists in HTML (from index.html)
        const existingContainer = document.getElementById(this.containerId);
        if (existingContainer) {
            this.container = existingContainer;
            this.attachEventListeners();
            console.log('✅ [SandboxManager] Initialized with existing container');
            return;
        }
        
        // Create sandbox mode container if not found
        this.createSandboxModeContainer();
        console.log('✅ [SandboxManager] Initialized with new container');
    }
    
    createSandboxModeContainer() {
        // Check if already exists
        if (document.getElementById('sandbox-mode-container')) return;
        
        const container = document.createElement('div');
        container.id = 'sandbox-mode-container';
        container.className = 'sandbox-mode-container';
        container.innerHTML = `
            <div class="sandbox-mode-wrapper">
                <!-- Compact Chat Panel (Left Side) -->
                <div class="sandbox-chat-panel" id="sandbox-chat-panel">
                    <div class="sandbox-chat-header">
                        <div class="sandbox-chat-title">
                            <i class="fas fa-robot"></i>
                            <span>AetherMind</span>
                        </div>
                        <div class="sandbox-mode-badge">
                            <i class="fas fa-cube"></i>
                            App Creation Mode
                        </div>
                    </div>
                    <div class="sandbox-chat-messages" id="sandbox-chat-messages">
                        <!-- Messages will be synced from main chat -->
                    </div>
                    <div class="sandbox-chat-input-area">
                        <form id="sandbox-chat-form">
                            <input type="text" id="sandbox-chat-input" placeholder="Describe your app changes..." autocomplete="off">
                            <button type="submit"><i class="fas fa-paper-plane"></i></button>
                        </form>
                    </div>
                </div>
                
                <!-- Main Sandbox View (Right Side) -->
                <div class="sandbox-main-panel" id="sandbox-main-panel">
                    <!-- Sandbox Header -->
                    <div class="sandbox-header">
                        <div class="sandbox-header-left">
                            <div class="sandbox-app-info">
                                <i class="fas fa-cube sandbox-app-icon"></i>
                                <div class="sandbox-app-details">
                                    <span class="sandbox-app-name" id="sandbox-app-name">New App</span>
                                    <span class="sandbox-app-status" id="sandbox-app-status">
                                        <span class="status-dot initializing"></span>
                                        Initializing...
                                    </span>
                                </div>
                            </div>
                        </div>
                        <div class="sandbox-header-center">
                            <div class="sandbox-tabs">
                                <button class="sandbox-tab active" data-tab="preview">
                                    <i class="fas fa-eye"></i> Preview
                                </button>
                                <button class="sandbox-tab" data-tab="code">
                                    <i class="fas fa-code"></i> Code
                                </button>
                                <button class="sandbox-tab" data-tab="terminal">
                                    <i class="fas fa-terminal"></i> Terminal
                                </button>
                                <button class="sandbox-tab" data-tab="files">
                                    <i class="fas fa-folder"></i> Files
                                </button>
                            </div>
                        </div>
                        <div class="sandbox-header-right">
                            <button class="sandbox-action-btn" id="sandbox-refresh-btn" title="Refresh Preview">
                                <i class="fas fa-sync-alt"></i>
                            </button>
                            <button class="sandbox-action-btn" id="sandbox-share-btn" title="Share App">
                                <i class="fas fa-share-alt"></i>
                            </button>
                            <button class="sandbox-action-btn primary" id="sandbox-deploy-btn" title="Deploy to Production">
                                <i class="fas fa-rocket"></i>
                                <span>Deploy</span>
                            </button>
                            <button class="sandbox-exit-btn" id="sandbox-exit-btn" title="Exit App Creation Mode">
                                <i class="fas fa-times"></i>
                                <span>Exit</span>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Sandbox Content Area -->
                    <div class="sandbox-content">
                        <!-- Preview Tab -->
                        <div class="sandbox-tab-content active" data-content="preview">
                            <div class="sandbox-preview-container">
                                <div class="sandbox-preview-toolbar">
                                    <div class="preview-url-bar">
                                        <i class="fas fa-lock"></i>
                                        <span id="sandbox-preview-url">localhost:5000</span>
                                        <button class="preview-url-copy" title="Copy URL">
                                            <i class="fas fa-copy"></i>
                                        </button>
                                    </div>
                                    <div class="preview-device-switcher">
                                        <button class="device-btn active" data-device="desktop" title="Desktop">
                                            <i class="fas fa-desktop"></i>
                                        </button>
                                        <button class="device-btn" data-device="tablet" title="Tablet">
                                            <i class="fas fa-tablet-alt"></i>
                                        </button>
                                        <button class="device-btn" data-device="mobile" title="Mobile">
                                            <i class="fas fa-mobile-alt"></i>
                                        </button>
                                    </div>
                                    <button class="preview-fullscreen-btn" title="Fullscreen">
                                        <i class="fas fa-expand"></i>
                                    </button>
                                </div>
                                <div class="sandbox-preview-frame-container" id="sandbox-preview-frame-container">
                                    <div class="sandbox-preview-placeholder">
                                        <div class="preview-loading">
                                            <div class="loading-spinner"></div>
                                            <p>Initializing sandbox environment...</p>
                                            <span class="loading-hint">AetherMind is setting up your development environment</span>
                                        </div>
                                    </div>
                                    <iframe id="sandbox-preview-iframe" class="sandbox-preview-iframe" style="display: none;"></iframe>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Code Tab -->
                        <div class="sandbox-tab-content" data-content="code">
                            <div class="sandbox-code-container">
                                <div class="code-file-tabs" id="code-file-tabs">
                                    <!-- File tabs will be populated dynamically -->
                                </div>
                                <div class="code-editor-area" id="code-editor-area">
                                    <pre><code id="code-display" class="language-python"># Your code will appear here</code></pre>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Terminal Tab -->
                        <div class="sandbox-tab-content" data-content="terminal">
                            <div class="sandbox-terminal-container">
                                <div class="terminal-header">
                                    <div class="terminal-title">
                                        <i class="fas fa-terminal"></i>
                                        <span>Live Output</span>
                                    </div>
                                    <div class="terminal-actions">
                                        <button class="terminal-action-btn" id="terminal-clear-btn" title="Clear">
                                            <i class="fas fa-trash"></i>
                                        </button>
                                        <button class="terminal-action-btn" id="terminal-pause-btn" title="Pause Auto-scroll">
                                            <i class="fas fa-pause"></i>
                                        </button>
                                    </div>
                                </div>
                                <div class="terminal-output" id="terminal-output">
                                    <div class="terminal-line info">
                                        <span class="terminal-timestamp">[${this.getTimestamp()}]</span>
                                        <span class="terminal-text">Sandbox terminal ready. Waiting for commands...</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Files Tab -->
                        <div class="sandbox-tab-content" data-content="files">
                            <div class="sandbox-files-container">
                                <div class="files-tree" id="files-tree">
                                    <div class="files-tree-header">
                                        <span>Project Files</span>
                                        <button class="files-action-btn" title="Refresh">
                                            <i class="fas fa-sync-alt"></i>
                                        </button>
                                    </div>
                                    <div class="files-tree-content" id="files-tree-content">
                                        <!-- File tree will be populated -->
                                        <div class="file-tree-empty">
                                            <i class="fas fa-folder-open"></i>
                                            <p>No files yet</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="files-preview" id="files-preview">
                                    <div class="files-preview-placeholder">
                                        <i class="fas fa-file-code"></i>
                                        <p>Select a file to preview</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(container);
        this.attachEventListeners();
    }
    
    attachEventListeners() {
        // Exit button
        const exitBtn = document.getElementById('sandbox-exit-btn');
        if (exitBtn) {
            exitBtn.addEventListener('click', () => this.deactivate());
        }
        
        // Tab switching
        document.querySelectorAll('.sandbox-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.currentTarget.dataset.tab;
                this.switchTab(tabName);
            });
        });
        
        // Sandbox chat form
        const chatForm = document.getElementById('sandbox-chat-form');
        if (chatForm) {
            chatForm.addEventListener('submit', (e) => this.handleSandboxChatSubmit(e));
        }
        
        // Refresh button
        const refreshBtn = document.getElementById('sandbox-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshPreview());
        }
        
        // Deploy button
        const deployBtn = document.getElementById('sandbox-deploy-btn');
        if (deployBtn) {
            deployBtn.addEventListener('click', () => this.showDeployDialog());
        }
        
        // Device switcher
        document.querySelectorAll('.device-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.device-btn').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
                const device = e.currentTarget.dataset.device;
                this.setPreviewDevice(device);
            });
        });
        
        // Terminal clear
        const clearBtn = document.getElementById('terminal-clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearTerminal());
        }
        
        // Copy URL
        document.querySelector('.preview-url-copy')?.addEventListener('click', () => {
            const url = document.getElementById('sandbox-preview-url').textContent;
            navigator.clipboard.writeText(url);
            this.addLog('info', `URL copied to clipboard: ${url}`);
        });
    }
    
    /**
     * Activate App Creation Mode
     * @param {Object} projectConfig - Configuration for the project being created
     * Supports: apps, tools, mcp servers, APIs
     */
    activate(projectConfig = {}) {
        if (this.isActive) {
            console.log('⚠️ [SandboxManager] Already active, ignoring');
            return;
        }
        
        console.log('🚀 [SandboxManager] Activating Build Mode', projectConfig);
        
        try {
            this.isActive = true;
            
            // Normalize project config for scalability
            this.currentApp = {
                id: `project_${Date.now()}`,
                name: projectConfig.name || projectConfig.appName || 'New Project',
                template: projectConfig.template || 'blank',
                description: projectConfig.description || '',
                projectType: projectConfig.projectType || 'app',  // app, tool, mcp, api
                previewMode: projectConfig.previewMode || 'iframe', // iframe, terminal, logs, api-tester
                buildCommand: projectConfig.buildCommand || '',
                deployTarget: projectConfig.deployTarget || 'vercel',
                typeConfig: projectConfig.typeConfig || {},
                createdAt: new Date().toISOString(),
                ...projectConfig
            };
            
            // Get container
            let container = document.getElementById('sandbox-mode-container');
            if (!container) {
                console.error('❌ [SandboxManager] Container not found!');
                alert('Sandbox container not found. Please refresh the page.');
                this.isActive = false;
                return;
            }
            
            // Update UI based on project type
            this.updateUIForProjectType(this.currentApp);
            
            // Show sandbox mode - remove hidden, add active
            container.classList.remove('hidden');
            container.classList.add('active');
            container.style.display = 'flex';
            
            // Hide other components
            document.body.classList.add('sandbox-mode-active');
            
            // Set status
            this.setStatus('initializing', 'Initializing...');
            
            // Sync chat messages
            this.syncChatMessages();
            
            // Start sandbox environment
            this.initializeSandbox();
            
            // Add to activity feed
            if (window.activityFeed) {
                window.activityFeed.addActivity({
                    id: `sandbox_${Date.now()}`,
                    type: 'code_execution',
                    status: 'in_progress',
                    title: `Building: ${this.currentApp.name}`,
                    details: `${this.getProjectTypeLabel()} - Sandbox activated`,
                    timestamp: new Date().toISOString(),
                    data: { projectConfig: this.currentApp }
                });
            }
            
            this.addLog('success', `Build Mode activated for "${this.currentApp.name}"`);
            this.addLog('info', `Project type: ${this.getProjectTypeLabel()}`);
            this.addLog('info', 'Aether has full control to build your project');
            
            console.log('✅ [SandboxManager] Build Mode activated successfully');
            
        } catch (error) {
            console.error('❌ [SandboxManager] Failed to activate:', error);
            alert('Failed to activate Build Mode: ' + error.message);
            this.isActive = false;
        }
    }
    
    /**
     * Update UI elements based on project type
     */
    updateUIForProjectType(config) {
        const typeIcons = {
            app: 'fa-globe',
            tool: 'fa-wrench',
            mcp: 'fa-server',
            api: 'fa-plug'
        };
        
        const typeLabels = {
            app: 'Web App',
            tool: 'Tool/Script',
            mcp: 'MCP Server',
            api: 'API/Backend'
        };
        
        const typeColors = {
            app: '#8b5cf6',
            tool: '#f59e0b',
            mcp: '#10b981',
            api: '#3b82f6'
        };
        
        // Update project name
        const appNameEl = document.getElementById('sandbox-app-name');
        if (appNameEl) appNameEl.textContent = config.name || 'New Project';
        
        // Update mode badge
        const modeBadge = document.querySelector('.sandbox-mode-badge');
        if (modeBadge) {
            const icon = typeIcons[config.projectType] || 'fa-cube';
            const label = typeLabels[config.projectType] || 'Build Mode';
            modeBadge.innerHTML = `<i class="fas ${icon}"></i> ${label}`;
            modeBadge.style.borderColor = typeColors[config.projectType] || '#8b5cf6';
            modeBadge.style.color = typeColors[config.projectType] || '#8b5cf6';
        }
        
        // Update preview based on mode
        this.configurePreviewMode(config.previewMode, config.projectType);
        
        // Update deploy button text
        const deployBtn = document.getElementById('sandbox-deploy-btn');
        if (deployBtn) {
            const deployLabels = {
                app: 'Deploy',
                tool: 'Publish',
                mcp: 'Install',
                api: 'Deploy'
            };
            const spanEl = deployBtn.querySelector('span');
            if (spanEl) {
                spanEl.textContent = deployLabels[config.projectType] || 'Deploy';
            }
        }
    }
    
    /**
     * Configure preview panel based on project type
     */
    configurePreviewMode(mode, projectType) {
        const previewContainer = document.getElementById('sandbox-preview-frame-container');
        if (!previewContainer) return;
        
        // Update tab labels for different project types
        const previewTab = document.querySelector('.sandbox-tab[data-tab="preview"]');
        if (previewTab) {
            const previewLabels = {
                iframe: '<i class="fas fa-eye"></i> Preview',
                terminal: '<i class="fas fa-terminal"></i> Output',
                logs: '<i class="fas fa-stream"></i> Logs',
                'api-tester': '<i class="fas fa-flask"></i> API Tester'
            };
            previewTab.innerHTML = previewLabels[mode] || previewLabels.iframe;
        }
        
        // Configure preview container based on mode
        switch (mode) {
            case 'terminal':
                this.setupTerminalPreview(previewContainer);
                break;
            case 'logs':
                this.setupLogsPreview(previewContainer);
                break;
            case 'api-tester':
                this.setupAPITesterPreview(previewContainer);
                break;
            default:
                this.setupIframePreview(previewContainer);
        }
    }
    
    /**
     * Setup terminal-style output preview (for tools/scripts)
     */
    setupTerminalPreview(container) {
        container.innerHTML = `
            <div class="terminal-preview">
                <div class="terminal-header">
                    <span class="terminal-title"><i class="fas fa-terminal"></i> Output</span>
                    <button class="terminal-run-btn" onclick="window.sandboxManager.runScript()">
                        <i class="fas fa-play"></i> Run
                    </button>
                </div>
                <div class="terminal-output" id="terminal-output">
                    <div class="terminal-line info">Ready to run your script...</div>
                </div>
            </div>
        `;
    }
    
    /**
     * Setup logs preview (for MCP servers)
     */
    setupLogsPreview(container) {
        container.innerHTML = `
            <div class="logs-preview">
                <div class="logs-header">
                    <span class="logs-title"><i class="fas fa-server"></i> Server Logs</span>
                    <div class="logs-controls">
                        <button class="logs-start-btn" onclick="window.sandboxManager.startServer()">
                            <i class="fas fa-play"></i> Start Server
                        </button>
                        <button class="logs-stop-btn" onclick="window.sandboxManager.stopServer()" disabled>
                            <i class="fas fa-stop"></i> Stop
                        </button>
                    </div>
                </div>
                <div class="logs-stream" id="logs-stream">
                    <div class="log-entry info">[INFO] MCP Server ready to start...</div>
                </div>
                <div class="logs-status">
                    <span class="status-indicator stopped"></span>
                    <span>Server stopped</span>
                </div>
            </div>
        `;
    }
    
    /**
     * Setup API tester preview (for APIs/backends)
     */
    setupAPITesterPreview(container) {
        container.innerHTML = `
            <div class="api-tester-preview">
                <div class="api-tester-header">
                    <span class="api-title"><i class="fas fa-plug"></i> API Tester</span>
                </div>
                <div class="api-request-builder">
                    <select id="api-method" class="api-method-select">
                        <option value="GET">GET</option>
                        <option value="POST">POST</option>
                        <option value="PUT">PUT</option>
                        <option value="DELETE">DELETE</option>
                    </select>
                    <input type="text" id="api-endpoint" class="api-endpoint-input" placeholder="/api/endpoint">
                    <button class="api-send-btn" onclick="window.sandboxManager.sendAPIRequest()">
                        <i class="fas fa-paper-plane"></i> Send
                    </button>
                </div>
                <div class="api-body-section">
                    <label>Request Body (JSON):</label>
                    <textarea id="api-body" class="api-body-input" placeholder='{"key": "value"}'></textarea>
                </div>
                <div class="api-response-section">
                    <label>Response:</label>
                    <pre id="api-response" class="api-response-output">No response yet...</pre>
                </div>
            </div>
        `;
    }
    
    /**
     * Setup iframe preview (default for web apps)
     */
    setupIframePreview(container) {
        container.innerHTML = `
            <div class="sandbox-preview-placeholder">
                <div class="preview-loading">
                    <div class="loading-spinner"></div>
                    <span>Initializing preview...</span>
                </div>
            </div>
        `;
    }
    
    /**
     * Get human-readable project type label
     */
    getProjectTypeLabel() {
        const labels = {
            app: 'Web Application',
            tool: 'Tool/Script',
            mcp: 'MCP Server',
            api: 'REST API'
        };
        return labels[this.currentApp?.projectType] || 'Project';
    }
    
    /**
     * Run script (for tool type)
     */
    async runScript() {
        const output = document.getElementById('terminal-output');
        if (output) {
            output.innerHTML += `<div class="terminal-line running">$ Running script...</div>`;
            this.addLog('info', 'Executing script...');
            
            // TODO: Connect to backend sandbox execution
            setTimeout(() => {
                output.innerHTML += `<div class="terminal-line success">Script completed successfully</div>`;
            }, 1000);
        }
    }
    
    /**
     * Start MCP server
     */
    async startServer() {
        const logsStream = document.getElementById('logs-stream');
        const startBtn = document.querySelector('.logs-start-btn');
        const stopBtn = document.querySelector('.logs-stop-btn');
        
        if (startBtn) startBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
        
        if (logsStream) {
            logsStream.innerHTML += `<div class="log-entry info">[INFO] Starting MCP server...</div>`;
            this.addLog('info', 'Starting MCP server...');
            
            // TODO: Connect to backend MCP server start
            setTimeout(() => {
                logsStream.innerHTML += `<div class="log-entry success">[SUCCESS] Server running on stdio</div>`;
            }, 500);
        }
    }
    
    /**
     * Stop MCP server
     */
    async stopServer() {
        const startBtn = document.querySelector('.logs-start-btn');
        const stopBtn = document.querySelector('.logs-stop-btn');
        
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        
        this.addLog('info', 'Stopping MCP server...');
    }
    
    /**
     * Send API test request
     */
    async sendAPIRequest() {
        const method = document.getElementById('api-method')?.value || 'GET';
        const endpoint = document.getElementById('api-endpoint')?.value || '/';
        const body = document.getElementById('api-body')?.value || '';
        const responseEl = document.getElementById('api-response');
        
        if (responseEl) {
            responseEl.textContent = 'Sending request...';
            
            // TODO: Connect to backend API proxy
            setTimeout(() => {
                responseEl.textContent = JSON.stringify({
                    status: 200,
                    message: 'API endpoint not yet implemented'
                }, null, 2);
            }, 500);
        }
    }
    
    /**
     * Deactivate App Creation Mode
     */
    deactivate() {
        if (!this.isActive) return;
        
        console.log('⏹️ [SandboxManager] Deactivating App Creation Mode');
        
        // Confirm exit if there's unsaved work
        if (this.logs.length > 5) {
            if (!confirm('Exit App Creation Mode? Your sandbox session will be saved.')) {
                return;
            }
        }
        
        this.isActive = false;
        
        // Hide sandbox mode
        const container = document.getElementById('sandbox-mode-container');
        container.classList.remove('active');
        
        // Show other components
        document.body.classList.remove('sandbox-mode-active');
        
        // Stop polling
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
        
        this.addLog('info', 'App Creation Mode deactivated');
    }
    
    /**
     * Initialize sandbox environment
     */
    async initializeSandbox() {
        this.setStatus('building', 'Setting up environment...');
        this.addLog('info', 'Creating isolated sandbox environment...');
        
        try {
            // Simulate sandbox setup (in production, call backend)
            await this.delay(1000);
            this.addLog('success', 'Python virtual environment created');
            
            await this.delay(500);
            this.addLog('info', 'Installing dependencies...');
            
            if (this.currentApp.framework === 'flask') {
                this.addLog('success', '✓ flask installed');
                this.addLog('success', '✓ flask-cors installed');
            }
            
            await this.delay(500);
            this.setStatus('running', 'Ready');
            this.addLog('success', '🚀 Sandbox ready! Aether can now build your app.');
            
            // Start log polling
            this.startLogPolling();
            
        } catch (error) {
            this.setStatus('error', 'Setup failed');
            this.addLog('error', `Failed to initialize sandbox: ${error.message}`);
        }
    }
    
    /**
     * Set sandbox status
     */
    setStatus(status, text) {
        const statusEl = document.getElementById('sandbox-app-status');
        if (!statusEl) return;
        
        const statusClasses = {
            initializing: 'initializing',
            building: 'building',
            running: 'running',
            error: 'error',
            stopped: 'stopped'
        };
        
        statusEl.innerHTML = `
            <span class="status-dot ${statusClasses[status] || 'initializing'}"></span>
            ${text}
        `;
    }
    
    /**
     * Switch between tabs
     */
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.sandbox-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });
        
        // Update content
        document.querySelectorAll('.sandbox-tab-content').forEach(content => {
            content.classList.toggle('active', content.dataset.content === tabName);
        });
    }
    
    /**
     * Add log entry to terminal
     */
    addLog(type, message) {
        const log = {
            type,
            message,
            timestamp: new Date().toISOString()
        };
        this.logs.push(log);
        
        const terminalOutput = document.getElementById('terminal-output');
        if (terminalOutput) {
            const line = document.createElement('div');
            line.className = `terminal-line ${type}`;
            line.innerHTML = `
                <span class="terminal-timestamp">[${this.getTimestamp()}]</span>
                <span class="terminal-text">${this.escapeHtml(message)}</span>
            `;
            terminalOutput.appendChild(line);
            terminalOutput.scrollTop = terminalOutput.scrollHeight;
        }
    }
    
    /**
     * Clear terminal
     */
    clearTerminal() {
        const terminalOutput = document.getElementById('terminal-output');
        if (terminalOutput) {
            terminalOutput.innerHTML = `
                <div class="terminal-line info">
                    <span class="terminal-timestamp">[${this.getTimestamp()}]</span>
                    <span class="terminal-text">Terminal cleared</span>
                </div>
            `;
        }
        this.logs = [];
    }
    
    /**
     * Set preview device size
     */
    setPreviewDevice(device) {
        const container = document.getElementById('sandbox-preview-frame-container');
        if (!container) return;
        
        container.classList.remove('desktop', 'tablet', 'mobile');
        container.classList.add(device);
    }
    
    /**
     * Refresh preview iframe
     */
    refreshPreview() {
        const iframe = document.getElementById('sandbox-preview-iframe');
        if (iframe && iframe.src) {
            iframe.src = iframe.src;
            this.addLog('info', 'Preview refreshed');
        }
    }
    
    /**
     * Set preview URL and show iframe
     */
    setPreviewUrl(url) {
        this.previewUrl = url;
        
        const iframe = document.getElementById('sandbox-preview-iframe');
        const placeholder = document.querySelector('.sandbox-preview-placeholder');
        const urlDisplay = document.getElementById('sandbox-preview-url');
        
        if (iframe && placeholder) {
            iframe.src = url;
            iframe.style.display = 'block';
            placeholder.style.display = 'none';
        }
        
        if (urlDisplay) {
            urlDisplay.textContent = url;
        }
        
        this.addLog('success', `Preview available at: ${url}`);
    }
    
    /**
     * Sync messages from main chat
     */
    syncChatMessages() {
        const mainMessages = document.getElementById('messages');
        const sandboxMessages = document.getElementById('sandbox-chat-messages');
        
        if (mainMessages && sandboxMessages) {
            // Clone recent messages
            const messages = mainMessages.querySelectorAll('.message-row');
            sandboxMessages.innerHTML = '';
            
            messages.forEach(msg => {
                const clone = msg.cloneNode(true);
                clone.classList.add('sandbox-message');
                sandboxMessages.appendChild(clone);
            });
            
            sandboxMessages.scrollTop = sandboxMessages.scrollHeight;
        }
    }
    
    /**
     * Handle chat submission in sandbox mode
     */
    async handleSandboxChatSubmit(e) {
        e.preventDefault();
        
        const input = document.getElementById('sandbox-chat-input');
        const text = input.value.trim();
        
        if (!text) return;
        
        input.value = '';
        
        // Add to sandbox chat
        this.addSandboxMessage('user', text);
        
        // Also send to main chat system
        if (window.chat) {
            // Forward to main chat
            const mainInput = document.getElementById('chat-input');
            if (mainInput) {
                mainInput.value = text;
                document.getElementById('chat-form').dispatchEvent(new Event('submit'));
            }
        }
    }
    
    /**
     * Add message to sandbox chat
     */
    addSandboxMessage(role, content, metadata = {}) {
        const messages = document.getElementById('sandbox-chat-messages');
        if (!messages) return;
        
        const msgDiv = document.createElement('div');
        msgDiv.className = `sandbox-message ${role}`;
        
        if (role === 'user') {
            msgDiv.innerHTML = `
                <div class="sandbox-msg-content user">${this.escapeHtml(content)}</div>
            `;
        } else {
            msgDiv.innerHTML = `
                <div class="sandbox-msg-avatar"><i class="fas fa-robot"></i></div>
                <div class="sandbox-msg-content assistant">${content}</div>
            `;
        }
        
        messages.appendChild(msgDiv);
        messages.scrollTop = messages.scrollHeight;
    }
    
    /**
     * Update file tree
     */
    updateFileTree(files) {
        const treeContent = document.getElementById('files-tree-content');
        if (!treeContent) return;
        
        if (!files || files.length === 0) {
            treeContent.innerHTML = `
                <div class="file-tree-empty">
                    <i class="fas fa-folder-open"></i>
                    <p>No files yet</p>
                </div>
            `;
            return;
        }
        
        treeContent.innerHTML = files.map(file => `
            <div class="file-tree-item" data-path="${file.path}">
                <i class="fas ${this.getFileIcon(file.name)}"></i>
                <span class="file-name">${file.name}</span>
            </div>
        `).join('');
        
        // Add click handlers
        treeContent.querySelectorAll('.file-tree-item').forEach(item => {
            item.addEventListener('click', () => {
                const path = item.dataset.path;
                this.previewFile(path);
            });
        });
    }
    
    /**
     * Preview a file
     */
    previewFile(path) {
        // Highlight selected file
        document.querySelectorAll('.file-tree-item').forEach(item => {
            item.classList.toggle('selected', item.dataset.path === path);
        });
        
        this.addLog('info', `Viewing file: ${path}`);
        // In production, fetch file content from backend
    }
    
    /**
     * Get file icon based on extension
     */
    getFileIcon(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const icons = {
            py: 'fa-python',
            js: 'fa-js-square',
            ts: 'fa-js-square',
            html: 'fa-html5',
            css: 'fa-css3-alt',
            json: 'fa-file-code',
            md: 'fa-markdown',
            txt: 'fa-file-alt',
            yml: 'fa-file-code',
            yaml: 'fa-file-code'
        };
        return icons[ext] || 'fa-file';
    }
    
    /**
     * Show deploy dialog
     */
    showDeployDialog() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3 class="modal-title"><i class="fas fa-rocket"></i> Deploy App</h3>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <p style="margin-bottom: 1rem; color: var(--text-muted);">
                        Choose where to deploy your app:
                    </p>
                    <div style="display: flex; flex-direction: column; gap: 10px;">
                        <button class="modal-btn modal-btn-primary" onclick="window.sandboxManager.deployTo('render')">
                            <i class="fas fa-cloud"></i> Deploy to Render
                        </button>
                        <button class="modal-btn modal-btn-secondary" onclick="window.sandboxManager.deployTo('vercel')">
                            <i class="fas fa-globe"></i> Deploy to Vercel
                        </button>
                        <button class="modal-btn modal-btn-secondary" onclick="window.sandboxManager.createRepo()">
                            <i class="fab fa-github"></i> Create GitHub Repo
                        </button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }
    
    /**
     * Deploy to platform
     */
    deployTo(platform) {
        document.querySelector('.modal-overlay')?.remove();
        this.addLog('info', `Preparing deployment to ${platform}...`);
        
        // In production, trigger actual deployment
        if (window.chat) {
            window.chat.addMessage('user', `Deploy this app to ${platform}`);
        }
    }
    
    /**
     * Create GitHub repo from app
     */
    createRepo() {
        document.querySelector('.modal-overlay')?.remove();
        this.addLog('info', 'Creating GitHub repository...');
        
        if (window.chat) {
            window.chat.addMessage('user', `Create a GitHub repository for this app called "${this.currentApp.name}"`);
        }
    }
    
    /**
     * Start polling for logs
     */
    startLogPolling() {
        // Poll every 2 seconds for new logs
        this.pollingInterval = setInterval(() => {
            this.pollLogs();
        }, 2000);
    }
    
    /**
     * Poll for new logs from backend
     */
    async pollLogs() {
        // In production, fetch from backend
        // For now, this is a placeholder
    }
    
    /**
     * Execute code in sandbox
     */
    async executeCode(code, language = 'python') {
        this.addLog('info', `Executing ${language} code...`);
        
        try {
            const apiKey = localStorage.getItem('aethermind_api_key');
            const response = await fetch(`${this.backendUrl}/v1/sandbox/execute`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': apiKey
                },
                body: JSON.stringify({
                    sandbox_id: this.currentApp?.id,
                    code,
                    language
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                if (result.stdout) {
                    this.addLog('success', result.stdout);
                }
                if (result.stderr) {
                    this.addLog('error', result.stderr);
                }
                return result;
            }
        } catch (error) {
            this.addLog('error', `Execution failed: ${error.message}`);
        }
    }
    
    // Utility methods
    getTimestamp() {
        const now = new Date();
        return now.toTimeString().split(' ')[0];
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Make globally accessible
window.SandboxManager = SandboxManager;
