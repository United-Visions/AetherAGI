// UIOrchestrator.js - Agent-controlled dynamic UI
// All UI changes flow through here, agent decides what to show

export class UIOrchestrator {
    constructor(shell) {
        this.shell = shell;
        this.api = null; // Set by shell after api init
        
        // Track what's currently visible
        this.activeComponents = new Set();
        
        // Lazy-loaded components
        this.components = {
            camera: null,
            voice: null,
            fileUploader: null,
            splitView: null,
            sandbox: null
        };
        
        // Component registry - all available but not loaded
        this.registry = {
            camera: {
                name: 'CameraCapture',
                path: '/static/js/components/CameraCapture.js',
                css: null
            },
            voice: {
                name: 'VoiceRecorder',
                path: '/static/js/components/VoiceRecorder.js',
                css: null
            },
            fileUploader: {
                name: 'FileUploader',
                path: '/static/js/components/FileUploader.js',
                css: null
            },
            splitView: {
                name: 'SplitViewPanel',
                path: '/static/js/components/SplitViewPanel.js',
                css: '/static/css/split-view.css'
            },
            sandbox: {
                name: 'SandboxManager',
                path: '/static/js/components/SandboxManager.js',
                css: '/static/css/sandbox-manager.css'
            },
            activityFeed: {
                name: 'ActivityFeed',
                path: '/static/js/components/ActivityFeed.js',
                css: '/static/css/activity-feed.css'
            },
            brainVisualizer: {
                name: 'BrainVisualizer',
                path: '/static/js/components/BrainVisualizer.js',
                css: '/static/css/brain-visualizer.css'
            },
            benchmarks: {
                name: 'GauntletDashboard',
                path: '/static/js/components/GauntletDashboard.js',
                css: '/static/css/benchmarks.css'
            }
        };
    }

    // Execute UI commands from agent
    async executeCommands(commands) {
        console.log('🎨 [UI] Executing commands:', commands);
        
        for (const cmd of commands) {
            try {
                await this.executeCommand(cmd);
            } catch (err) {
                console.error(`Failed to execute command ${cmd.action}:`, err);
            }
        }
    }

    async executeCommand(cmd) {
        switch (cmd.action) {
            case 'show_component':
                await this.showComponent(cmd.component, cmd.options);
                break;
            case 'hide_component':
                this.hideComponent(cmd.component);
                break;
            case 'inject_panel':
                this.injectPanel(cmd.html, cmd.position);
                break;
            case 'remove_panel':
                this.removePanel(cmd.panelId);
                break;
            case 'update_header':
                this.updateHeader(cmd.options);
                break;
            case 'show_quick_actions':
                this.shell.elements.quickActions.classList.remove('hidden');
                break;
            case 'hide_quick_actions':
                this.shell.elements.quickActions.classList.add('hidden');
                break;
            case 'inject_css':
                this.injectCSS(cmd.css);
                break;
            case 'show_toast':
                this.shell.toast(cmd.message, cmd.type);
                break;
            case 'request_media':
                await this.requestMedia(cmd.type, cmd.reason);
                break;
            case 'start_task':
                this.shell.tasks.addTask(cmd.task);
                break;
            case 'forge_tool':
                await this.forgeTool(cmd.spec);
                break;
            case 'add_mcp_server':
                await this.addMCPServer(cmd.config);
                break;
            default:
                console.warn('Unknown UI command:', cmd.action);
        }
    }

    // Lazy load and show a component
    async showComponent(componentId, options = {}) {
        console.log(`📦 [UI] Loading component: ${componentId}`);
        
        const reg = this.registry[componentId];
        if (!reg) {
            console.warn(`Component not found: ${componentId}`);
            return;
        }
        
        // Load CSS if needed
        if (reg.css && !document.querySelector(`link[href="${reg.css}"]`)) {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = reg.css;
            document.head.appendChild(link);
        }
        
        // Load JS module if not already loaded
        if (!this.components[componentId]) {
            try {
                const module = await import(reg.path);
                const ComponentClass = module[reg.name] || module.default;
                
                // Create container if needed
                let containerId = `${componentId}-container`;
                let container = document.getElementById(containerId);
                if (!container) {
                    container = document.createElement('div');
                    container.id = containerId;
                    this.shell.elements.dynamicSlots.appendChild(container);
                }
                
                this.components[componentId] = new ComponentClass(containerId, options);
            } catch (err) {
                console.error(`Failed to load component ${componentId}:`, err);
                return;
            }
        }
        
        this.activeComponents.add(componentId);
        this.shell.elements.dynamicSlots.classList.remove('hidden');
    }

    hideComponent(componentId) {
        this.activeComponents.delete(componentId);
        
        const container = document.getElementById(`${componentId}-container`);
        if (container) {
            container.remove();
        }
        
        // Hide dynamic slots if empty
        if (this.activeComponents.size === 0) {
            this.shell.elements.dynamicSlots.classList.add('hidden');
        }
    }

    // Inject raw HTML panel (for agent-generated UI)
    injectPanel(html, position = 'dynamic') {
        const panelId = `panel_${Date.now()}`;
        const panel = document.createElement('div');
        panel.id = panelId;
        panel.className = 'injected-panel';
        panel.innerHTML = html;
        
        if (position === 'dynamic') {
            this.shell.elements.dynamicSlots.appendChild(panel);
            this.shell.elements.dynamicSlots.classList.remove('hidden');
        } else if (position === 'message') {
            // Insert as a message
            this.shell.elements.messagesContainer.appendChild(panel);
            this.shell.scrollToBottom();
        }
        
        return panelId;
    }

    removePanel(panelId) {
        document.getElementById(panelId)?.remove();
    }

    updateHeader(options) {
        if (options.brandTag) {
            const tag = document.querySelector('.brand-tag');
            if (tag) tag.textContent = options.brandTag;
        }
        if (options.brandName) {
            const name = document.querySelector('.brand-name');
            if (name) name.textContent = options.brandName;
        }
    }

    injectCSS(css) {
        const style = document.createElement('style');
        style.textContent = css;
        document.head.appendChild(style);
    }

    // Request media from user (photo/video)
    async requestMedia(type, reason) {
        console.log(`📷 [UI] Requesting ${type} for: ${reason}`);
        
        // Lazy load camera component
        if (!this.components.camera) {
            await this.showComponent('camera');
        }
        
        // Show quick actions with camera highlighted
        this.shell.elements.quickActions.classList.remove('hidden');
        
        // Flash the camera button to draw attention
        const cameraBtn = document.getElementById('camera-btn');
        if (cameraBtn) {
            cameraBtn.classList.add('highlight');
            setTimeout(() => cameraBtn.classList.remove('highlight'), 3000);
        }
    }

    // Activate camera (called from button)
    async activateCamera() {
        if (!this.components.camera) {
            try {
                const module = await import('/static/js/components/CameraCapture.js');
                this.components.camera = new module.CameraCapture('camera-btn', (file) => {
                    this.handleMediaCapture(file);
                });
            } catch (err) {
                console.error('Failed to load camera:', err);
                this.shell.toast('Camera not available', 'error');
                return;
            }
        }
        
        // Camera component handles its own activation
        this.components.camera.open?.();
    }

    // Activate voice (called from button)
    async activateVoice() {
        if (!this.components.voice) {
            try {
                const module = await import('/static/js/components/VoiceRecorder.js');
                this.components.voice = new module.VoiceRecorder('voice-btn', (file) => {
                    this.handleMediaCapture(file);
                });
            } catch (err) {
                console.error('Failed to load voice recorder:', err);
                this.shell.toast('Voice recording not available', 'error');
                return;
            }
        }
    }

    handleMediaCapture(file) {
        console.log('📁 [UI] Media captured:', file.name);
        
        // Add to file previews
        this.shell.addFilePreview(file);
        this.shell.elements.filePreviews.classList.remove('hidden');
        
        this.shell.toast(`${file.type.startsWith('image') ? 'Photo' : 'Recording'} captured!`, 'success');
    }

    // Tool forging - create new capabilities
    async forgeTool(spec) {
        console.log('🔧 [UI] Forging tool:', spec);
        
        this.shell.toast(`Creating tool: ${spec.name}...`, 'info');
        
        // This creates a background task
        this.shell.tasks.addTask({
            id: `forge_${Date.now()}`,
            type: 'tool_forge',
            name: `Forging: ${spec.name}`,
            status: 'running',
            spec: spec
        });
        
        // Actual forging happens on backend
        try {
            const result = await fetch('/v1/tools/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': localStorage.getItem('aethermind_api_key')
                },
                body: JSON.stringify(spec)
            });
            
            if (result.ok) {
                this.shell.toast(`Tool "${spec.name}" created!`, 'success');
            } else {
                throw new Error('Tool creation failed');
            }
        } catch (err) {
            this.shell.toast(`Failed to create tool: ${err.message}`, 'error');
        }
    }

    // Add MCP server dynamically
    async addMCPServer(config) {
        console.log('🖥️ [UI] Adding MCP server:', config);
        
        this.shell.tasks.addTask({
            id: `mcp_${Date.now()}`,
            type: 'mcp_setup',
            name: `Setting up: ${config.name}`,
            status: 'running',
            config: config
        });
        
        try {
            const result = await fetch('/v1/mcp/add', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': localStorage.getItem('aethermind_api_key')
                },
                body: JSON.stringify(config)
            });
            
            if (result.ok) {
                this.shell.toast(`MCP server "${config.name}" connected!`, 'success');
            }
        } catch (err) {
            this.shell.toast(`Failed to add MCP server: ${err.message}`, 'error');
        }
    }
}
