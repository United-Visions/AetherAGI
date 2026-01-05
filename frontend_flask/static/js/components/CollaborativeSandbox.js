// components/CollaborativeSandbox.js - Live Code Collaboration & Execution Environment

export class CollaborativeSandbox {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.currentFile = null;
        this.files = new Map();
        this.executionResults = [];
        this.isCollaborating = false;
        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="sandbox-container">
                <div class="sandbox-header">
                    <div class="sandbox-tabs">
                        <button class="sandbox-tab active" data-tab="editor">
                            <i class="fas fa-code"></i> Editor
                        </button>
                        <button class="sandbox-tab" data-tab="terminal">
                            <i class="fas fa-terminal"></i> Terminal
                        </button>
                        <button class="sandbox-tab" data-tab="output">
                            <i class="fas fa-chart-bar"></i> Output
                        </button>
                        <button class="sandbox-tab" data-tab="files">
                            <i class="fas fa-folder-tree"></i> Files
                        </button>
                    </div>
                    <div class="sandbox-actions">
                        <button class="sandbox-btn" id="sandbox-run" title="Run Code">
                            <i class="fas fa-play"></i> Run
                        </button>
                        <button class="sandbox-btn" id="sandbox-save" title="Save">
                            <i class="fas fa-save"></i> Save
                        </button>
                        <button class="sandbox-btn" id="sandbox-collaborate" title="Toggle Collaboration">
                            <i class="fas fa-users"></i> Collaborate
                        </button>
                        <button class="sandbox-btn" id="sandbox-fullscreen" title="Fullscreen">
                            <i class="fas fa-expand"></i>
                        </button>
                    </div>
                </div>
                
                <div class="sandbox-content">
                    <!-- Editor Tab -->
                    <div class="sandbox-panel active" data-panel="editor">
                        <div class="editor-sidebar">
                            <div class="file-tree" id="file-tree"></div>
                            <button class="add-file-btn" id="add-file">
                                <i class="fas fa-plus"></i> New File
                            </button>
                        </div>
                        <div class="editor-main">
                            <div class="editor-toolbar">
                                <span class="current-file-name" id="current-file-name">untitled.py</span>
                                <div class="editor-indicators">
                                    <span class="collab-cursor" id="aether-cursor" style="display: none;">
                                        <i class="fas fa-robot"></i> AetherMind is editing...
                                    </span>
                                    <span class="syntax-indicator">
                                        <i class="fas fa-check-circle"></i> No errors
                                    </span>
                                </div>
                            </div>
                            <div class="code-editor" id="code-editor" contenteditable="true" spellcheck="false"></div>
                            <div class="editor-footer">
                                <span>Line 1, Col 1</span>
                                <span>Python</span>
                                <span>UTF-8</span>
                            </div>
                        </div>
                    </div>

                    <!-- Terminal Tab -->
                    <div class="sandbox-panel" data-panel="terminal">
                        <div class="terminal-container">
                            <div class="terminal-output" id="terminal-output">
                                <div class="terminal-line">$ Ready to execute commands...</div>
                            </div>
                            <div class="terminal-input-container">
                                <span class="terminal-prompt">$</span>
                                <input type="text" class="terminal-input" id="terminal-input" placeholder="Enter command...">
                            </div>
                        </div>
                    </div>

                    <!-- Output Tab -->
                    <div class="sandbox-panel" data-panel="output">
                        <div class="output-container" id="output-container">
                            <div class="output-empty">
                                <i class="fas fa-play-circle"></i>
                                <p>Run your code to see output here</p>
                            </div>
                        </div>
                    </div>

                    <!-- Files Tab -->
                    <div class="sandbox-panel" data-panel="files">
                        <div class="files-browser">
                            <div class="files-header">
                                <h3>Workspace Files</h3>
                                <input type="text" class="files-search" placeholder="Search files...">
                            </div>
                            <div class="files-list" id="files-list"></div>
                        </div>
                    </div>
                </div>

                <!-- Execution Status Overlay -->
                <div class="execution-overlay" id="execution-overlay" style="display: none;">
                    <div class="execution-status">
                        <div class="spinner"></div>
                        <p>Executing code...</p>
                    </div>
                </div>
            </div>
        `;

        this.setupEditor();
        this.setupTabs();
        this.setupActions();
        this.setupFileTree();
        this.setupTerminal();
    }

    setupEditor() {
        this.editor = document.getElementById('code-editor');
        
        // Syntax highlighting on input
        this.editor.addEventListener('input', () => {
            this.applySyntaxHighlighting();
            this.updateCursorPosition();
        });

        // Prevent default tab behavior
        this.editor.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                e.preventDefault();
                document.execCommand('insertText', false, '    ');
            }
        });

        // Load default code
        this.loadFile('untitled.py', '# Welcome to AetherMind Sandbox\n# Write your code here\n\ndef main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()');
    }

    setupTabs() {
        const tabs = this.container.querySelectorAll('.sandbox-tab');
        const panels = this.container.querySelectorAll('.sandbox-panel');

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetPanel = tab.dataset.tab;
                
                tabs.forEach(t => t.classList.remove('active'));
                panels.forEach(p => p.classList.remove('active'));
                
                tab.classList.add('active');
                const panel = this.container.querySelector(`[data-panel="${targetPanel}"]`);
                if (panel) panel.classList.add('active');
            });
        });
    }

    setupActions() {
        document.getElementById('sandbox-run').addEventListener('click', () => {
            this.executeCode();
        });

        document.getElementById('sandbox-save').addEventListener('click', () => {
            this.saveFile();
        });

        document.getElementById('sandbox-collaborate').addEventListener('click', () => {
            this.toggleCollaboration();
        });

        document.getElementById('sandbox-fullscreen').addEventListener('click', () => {
            this.toggleFullscreen();
        });

        document.getElementById('add-file').addEventListener('click', () => {
            this.createNewFile();
        });
    }

    setupFileTree() {
        this.fileTree = document.getElementById('file-tree');
        this.updateFileTree();
    }

    setupTerminal() {
        const terminalInput = document.getElementById('terminal-input');
        const terminalOutput = document.getElementById('terminal-output');

        terminalInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const command = terminalInput.value.trim();
                if (command) {
                    this.executeTerminalCommand(command);
                    terminalInput.value = '';
                }
            }
        });
    }

    loadFile(filename, content) {
        this.files.set(filename, content);
        this.currentFile = filename;
        this.editor.textContent = content;
        document.getElementById('current-file-name').textContent = filename;
        this.applySyntaxHighlighting();
        this.updateFileTree();
    }

    saveFile() {
        if (this.currentFile) {
            const content = this.editor.textContent;
            this.files.set(this.currentFile, content);
            this.showNotification('File saved successfully', 'success');
        }
    }

    createNewFile() {
        const filename = prompt('Enter filename:', 'new_file.py');
        if (filename) {
            this.loadFile(filename, '# New file\n');
        }
    }

    updateFileTree() {
        this.fileTree.innerHTML = '';
        
        for (const [filename, content] of this.files) {
            const fileItem = document.createElement('div');
            fileItem.className = `file-item ${filename === this.currentFile ? 'active' : ''}`;
            
            const icon = this.getFileIcon(filename);
            fileItem.innerHTML = `
                <i class="${icon}"></i>
                <span>${filename}</span>
                <span class="file-size">${this.formatBytes(content.length)}</span>
            `;
            
            fileItem.addEventListener('click', () => {
                this.loadFile(filename, this.files.get(filename));
            });
            
            this.fileTree.appendChild(fileItem);
        }
    }

    async executeCode() {
        const code = this.editor.textContent;
        const overlay = document.getElementById('execution-overlay');
        const outputContainer = document.getElementById('output-container');
        
        overlay.style.display = 'flex';
        
        try {
            // Show Aether is working
            if (this.isCollaborating) {
                this.showAetherCursor();
            }

            // Call backend execution API
            const response = await fetch('/api/sandbox/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `ApiKey ${localStorage.getItem('aethermind_api_key')}`
                },
                body: JSON.stringify({
                    code: code,
                    language: this.detectLanguage(this.currentFile),
                    filename: this.currentFile
                })
            });

            const result = await response.json();
            
            // Display output
            outputContainer.innerHTML = `
                <div class="execution-result ${result.success ? 'success' : 'error'}">
                    <div class="execution-header">
                        <i class="fas fa-${result.success ? 'check-circle' : 'exclamation-triangle'}"></i>
                        <span>${result.success ? 'Execution Successful' : 'Execution Failed'}</span>
                        <span class="execution-time">${result.execution_time}ms</span>
                    </div>
                    ${result.stdout ? `
                        <div class="execution-section">
                            <h4>Standard Output</h4>
                            <pre><code>${this.escapeHtml(result.stdout)}</code></pre>
                        </div>
                    ` : ''}
                    ${result.stderr ? `
                        <div class="execution-section error">
                            <h4>Error Output</h4>
                            <pre><code>${this.escapeHtml(result.stderr)}</code></pre>
                        </div>
                    ` : ''}
                    ${result.return_value !== undefined ? `
                        <div class="execution-section">
                            <h4>Return Value</h4>
                            <pre><code>${this.escapeHtml(JSON.stringify(result.return_value, null, 2))}</code></pre>
                        </div>
                    ` : ''}
                </div>
            `;

            // Switch to output tab
            this.container.querySelector('[data-tab="output"]').click();
            
            this.executionResults.push(result);
            
        } catch (error) {
            outputContainer.innerHTML = `
                <div class="execution-result error">
                    <div class="execution-header">
                        <i class="fas fa-exclamation-triangle"></i>
                        <span>Execution Error</span>
                    </div>
                    <div class="execution-section error">
                        <pre><code>${error.message}</code></pre>
                    </div>
                </div>
            `;
        } finally {
            overlay.style.display = 'none';
            this.hideAetherCursor();
        }
    }

    async executeTerminalCommand(command) {
        const terminalOutput = document.getElementById('terminal-output');
        
        // Add command to output
        const commandLine = document.createElement('div');
        commandLine.className = 'terminal-line command';
        commandLine.textContent = `$ ${command}`;
        terminalOutput.appendChild(commandLine);

        try {
            const response = await fetch('/api/sandbox/terminal', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `ApiKey ${localStorage.getItem('aethermind_api_key')}`
                },
                body: JSON.stringify({ command })
            });

            const result = await response.json();
            
            const outputLine = document.createElement('div');
            outputLine.className = `terminal-line ${result.success ? '' : 'error'}`;
            outputLine.textContent = result.output || result.error;
            terminalOutput.appendChild(outputLine);
            
        } catch (error) {
            const errorLine = document.createElement('div');
            errorLine.className = 'terminal-line error';
            errorLine.textContent = `Error: ${error.message}`;
            terminalOutput.appendChild(errorLine);
        }

        terminalOutput.scrollTop = terminalOutput.scrollHeight;
    }

    toggleCollaboration() {
        this.isCollaborating = !this.isCollaborating;
        const btn = document.getElementById('sandbox-collaborate');
        
        if (this.isCollaborating) {
            btn.classList.add('active');
            this.showNotification('Collaboration mode enabled - AetherMind can now edit code', 'info');
        } else {
            btn.classList.remove('active');
            this.showNotification('Collaboration mode disabled', 'info');
        }
    }

    showAetherCursor() {
        const cursor = document.getElementById('aether-cursor');
        cursor.style.display = 'flex';
        
        // Simulate Aether typing
        setTimeout(() => {
            cursor.innerHTML = '<i class="fas fa-robot"></i> AetherMind suggests...';
        }, 1000);
    }

    hideAetherCursor() {
        const cursor = document.getElementById('aether-cursor');
        cursor.style.display = 'none';
    }

    applySyntaxHighlighting() {
        // Simple syntax highlighting (can be enhanced with a library like Prism.js)
        let content = this.editor.textContent;
        
        // This is a simplified version - in production use a proper highlighter
        // For now, just preserve the text content
        // In a real implementation, you'd wrap keywords, strings, etc. in spans
    }

    updateCursorPosition() {
        // Update cursor position indicator
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            // Calculate line and column
            // This is simplified - real implementation would be more robust
        }
    }

    toggleFullscreen() {
        this.container.classList.toggle('sandbox-fullscreen');
        const icon = document.querySelector('#sandbox-fullscreen i');
        icon.className = this.container.classList.contains('sandbox-fullscreen') 
            ? 'fas fa-compress' 
            : 'fas fa-expand';
    }

    detectLanguage(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const languages = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'go': 'go',
            'rs': 'rust',
            'rb': 'ruby',
            'php': 'php'
        };
        return languages[ext] || 'python';
    }

    getFileIcon(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const icons = {
            'py': 'fab fa-python',
            'js': 'fab fa-js-square',
            'html': 'fab fa-html5',
            'css': 'fab fa-css3-alt',
            'json': 'fas fa-brackets-curly',
            'md': 'fab fa-markdown'
        };
        return icons[ext] || 'fas fa-file-code';
    }

    formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `sandbox-notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Public API for AetherMind to edit code
    aetherEdit(changes) {
        if (!this.isCollaborating) return;
        
        this.showAetherCursor();
        
        // Apply changes gradually with animation
        setTimeout(() => {
            const currentContent = this.editor.textContent;
            this.editor.textContent = changes.newContent;
            this.applySyntaxHighlighting();
            this.hideAetherCursor();
        }, 500);
    }
}
