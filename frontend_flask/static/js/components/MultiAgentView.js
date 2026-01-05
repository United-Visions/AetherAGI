// components/MultiAgentView.js - Multi-Agent Collaboration Visualization

export class MultiAgentView {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.agents = new Map();
        this.communications = [];
        this.consensusHistory = [];
        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="multi-agent-container">
                <div class="agent-header">
                    <h3>
                        <i class="fas fa-users"></i>
                        Multi-Agent System
                    </h3>
                    <div class="agent-controls">
                        <button class="agent-btn" id="agent-add">
                            <i class="fas fa-plus"></i> Add Agent
                        </button>
                        <button class="agent-btn" id="agent-consensus">
                            <i class="fas fa-handshake"></i> View Consensus
                        </button>
                    </div>
                </div>

                <div class="agent-grid" id="agent-grid"></div>

                <div class="communication-flow" id="communication-flow">
                    <div class="flow-header">
                        <h4>Agent Communication Flow</h4>
                        <button class="flow-clear" id="flow-clear">
                            <i class="fas fa-trash"></i> Clear
                        </button>
                    </div>
                    <canvas id="communication-canvas"></canvas>
                    <div class="flow-legend">
                        <div class="flow-legend-item">
                            <div class="flow-line" style="background: #10b981"></div>
                            <span>Agreement</span>
                        </div>
                        <div class="flow-legend-item">
                            <div class="flow-line" style="background: #f59e0b"></div>
                            <span>Discussion</span>
                        </div>
                        <div class="flow-legend-item">
                            <div class="flow-line" style="background: #ef4444"></div>
                            <span>Disagreement</span>
                        </div>
                    </div>
                </div>

                <div class="handoff-tracker" id="handoff-tracker">
                    <h4>Control Handoff History</h4>
                    <div class="handoff-timeline" id="handoff-timeline"></div>
                </div>
            </div>
        `;

        this.setupCanvas();
        this.setupControls();
        this.initializeDefaultAgents();
        this.startAnimation();
    }

    setupCanvas() {
        this.canvas = document.getElementById('communication-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        const resize = () => {
            this.canvas.width = this.canvas.offsetWidth * window.devicePixelRatio;
            this.canvas.height = 300 * window.devicePixelRatio;
            this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        };
        resize();
        window.addEventListener('resize', resize);
    }

    setupControls() {
        document.getElementById('agent-add').addEventListener('click', () => {
            this.showAddAgentModal();
        });

        document.getElementById('agent-consensus').addEventListener('click', () => {
            this.showConsensusModal();
        });

        document.getElementById('flow-clear').addEventListener('click', () => {
            this.communications = [];
        });
    }

    initializeDefaultAgents() {
        // Initialize domain-specific agents
        this.addAgent({
            id: 'code_specialist',
            name: 'Code Specialist',
            domain: 'code',
            color: '#8b5cf6',
            icon: 'fas fa-code',
            status: 'idle',
            expertise: ['Programming', 'Debugging', 'Architecture'],
            confidence: 0.95
        });

        this.addAgent({
            id: 'research_specialist',
            name: 'Research Specialist',
            domain: 'research',
            color: '#3b82f6',
            icon: 'fas fa-microscope',
            status: 'idle',
            expertise: ['Analysis', 'Literature Review', 'Data'],
            confidence: 0.92
        });

        this.addAgent({
            id: 'business_specialist',
            name: 'Business Specialist',
            domain: 'business',
            color: '#10b981',
            icon: 'fas fa-briefcase',
            status: 'idle',
            expertise: ['Strategy', 'Finance', 'Marketing'],
            confidence: 0.88
        });

        this.addAgent({
            id: 'orchestrator',
            name: 'Orchestrator',
            domain: 'general',
            color: '#f59e0b',
            icon: 'fas fa-brain',
            status: 'active',
            expertise: ['Coordination', 'Meta-reasoning', 'Synthesis'],
            confidence: 1.0
        });
    }

    addAgent(agentData) {
        this.agents.set(agentData.id, agentData);
        this.renderAgentCard(agentData);
    }

    renderAgentCard(agent) {
        const grid = document.getElementById('agent-grid');
        
        const card = document.createElement('div');
        card.className = `agent-card agent-status-${agent.status}`;
        card.id = `agent-card-${agent.id}`;
        card.style.borderColor = agent.color;
        
        card.innerHTML = `
            <div class="agent-card-header" style="background: ${agent.color}">
                <div class="agent-avatar">
                    <i class="${agent.icon}"></i>
                </div>
                <div class="agent-info">
                    <div class="agent-name">${agent.name}</div>
                    <div class="agent-domain">${agent.domain}</div>
                </div>
                <div class="agent-status-indicator">
                    <div class="status-dot ${agent.status}"></div>
                </div>
            </div>
            
            <div class="agent-card-body">
                <div class="agent-section">
                    <h5>Expertise</h5>
                    <div class="agent-tags">
                        ${agent.expertise.map(e => `<span class="agent-tag">${e}</span>`).join('')}
                    </div>
                </div>
                
                <div class="agent-section">
                    <h5>Confidence</h5>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${agent.confidence * 100}%; background: ${agent.color}"></div>
                        <span class="confidence-value">${(agent.confidence * 100).toFixed(0)}%</span>
                    </div>
                </div>
                
                <div class="agent-section">
                    <h5>Current Task</h5>
                    <div class="agent-task" id="agent-task-${agent.id}">
                        ${agent.status === 'idle' ? 'Waiting for assignment' : 'Processing...'}
                    </div>
                </div>
            </div>
            
            <div class="agent-card-footer">
                <button class="agent-action-btn" onclick="multiAgentView.activateAgent('${agent.id}')">
                    <i class="fas fa-play-circle"></i> Activate
                </button>
                <button class="agent-action-btn" onclick="multiAgentView.viewAgentHistory('${agent.id}')">
                    <i class="fas fa-history"></i> History
                </button>
            </div>
        `;
        
        grid.appendChild(card);
    }

    updateAgentStatus(agentId, status, task = null) {
        const agent = this.agents.get(agentId);
        if (!agent) return;
        
        agent.status = status;
        
        const card = document.getElementById(`agent-card-${agentId}`);
        if (card) {
            card.className = `agent-card agent-status-${status}`;
            
            const statusDot = card.querySelector('.status-dot');
            statusDot.className = `status-dot ${status}`;
            
            if (task) {
                const taskElement = document.getElementById(`agent-task-${agentId}`);
                if (taskElement) {
                    taskElement.textContent = task;
                }
            }
        }
    }

    addCommunication(fromAgentId, toAgentId, message, agreementLevel) {
        const fromAgent = this.agents.get(fromAgentId);
        const toAgent = this.agents.get(toAgentId);
        
        if (!fromAgent || !toAgent) return;
        
        const communication = {
            id: Date.now(),
            from: fromAgent,
            to: toAgent,
            message: message,
            agreementLevel: agreementLevel, // 0-1
            timestamp: new Date(),
            active: true
        };
        
        this.communications.push(communication);
        
        // Fade out after 5 seconds
        setTimeout(() => {
            communication.active = false;
        }, 5000);
        
        // Log handoff if control is transferred
        if (message.includes('handoff') || message.includes('transfer')) {
            this.logHandoff(fromAgentId, toAgentId, message);
        }
    }

    logHandoff(fromAgentId, toAgentId, reason) {
        const timeline = document.getElementById('handoff-timeline');
        
        const fromAgent = this.agents.get(fromAgentId);
        const toAgent = this.agents.get(toAgentId);
        
        const handoffItem = document.createElement('div');
        handoffItem.className = 'handoff-item fade-in';
        handoffItem.innerHTML = `
            <div class="handoff-time">${new Date().toLocaleTimeString()}</div>
            <div class="handoff-flow">
                <div class="handoff-agent" style="border-color: ${fromAgent.color}">
                    <i class="${fromAgent.icon}"></i>
                    ${fromAgent.name}
                </div>
                <div class="handoff-arrow">
                    <i class="fas fa-arrow-right"></i>
                </div>
                <div class="handoff-agent" style="border-color: ${toAgent.color}">
                    <i class="${toAgent.icon}"></i>
                    ${toAgent.name}
                </div>
            </div>
            <div class="handoff-reason">${reason}</div>
        `;
        
        timeline.insertBefore(handoffItem, timeline.firstChild);
        
        // Keep only last 10
        while (timeline.children.length > 10) {
            timeline.removeChild(timeline.lastChild);
        }
    }

    startAnimation() {
        const animate = () => {
            this.renderCommunications();
            requestAnimationFrame(animate);
        };
        animate();
    }

    renderCommunications() {
        const ctx = this.ctx;
        const width = this.canvas.offsetWidth;
        const height = this.canvas.offsetHeight / window.devicePixelRatio;
        
        ctx.clearRect(0, 0, width, height);
        
        // Position agents in a circle
        const agentPositions = new Map();
        const agents = Array.from(this.agents.values());
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) / 3;
        
        agents.forEach((agent, i) => {
            const angle = (i / agents.length) * Math.PI * 2 - Math.PI / 2;
            const x = centerX + Math.cos(angle) * radius;
            const y = centerY + Math.sin(angle) * radius;
            agentPositions.set(agent.id, { x, y });
        });
        
        // Draw active communications
        this.communications.filter(c => c.active).forEach(comm => {
            const fromPos = agentPositions.get(comm.from.id);
            const toPos = agentPositions.get(comm.to.id);
            
            if (!fromPos || !toPos) return;
            
            // Color based on agreement level
            let color;
            if (comm.agreementLevel > 0.7) color = '#10b981'; // Green - agreement
            else if (comm.agreementLevel > 0.4) color = '#f59e0b'; // Yellow - discussion
            else color = '#ef4444'; // Red - disagreement
            
            // Animated flow
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.globalAlpha = 0.6;
            
            ctx.beginPath();
            ctx.moveTo(fromPos.x, fromPos.y);
            
            // Quadratic curve for nice flow
            const midX = (fromPos.x + toPos.x) / 2;
            const midY = (fromPos.y + toPos.y) / 2 - 30;
            ctx.quadraticCurveTo(midX, midY, toPos.x, toPos.y);
            ctx.stroke();
            
            // Draw arrowhead
            const angle = Math.atan2(toPos.y - midY, toPos.x - midX);
            const arrowSize = 10;
            ctx.beginPath();
            ctx.moveTo(toPos.x, toPos.y);
            ctx.lineTo(
                toPos.x - arrowSize * Math.cos(angle - Math.PI / 6),
                toPos.y - arrowSize * Math.sin(angle - Math.PI / 6)
            );
            ctx.lineTo(
                toPos.x - arrowSize * Math.cos(angle + Math.PI / 6),
                toPos.y - arrowSize * Math.sin(angle + Math.PI / 6)
            );
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
            
            ctx.globalAlpha = 1;
        });
        
        // Draw agent nodes
        agents.forEach(agent => {
            const pos = agentPositions.get(agent.id);
            if (!pos) return;
            
            // Node background
            ctx.fillStyle = agent.color;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);
            ctx.fill();
            
            // Status indicator
            if (agent.status === 'active') {
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 25, 0, Math.PI * 2);
                ctx.stroke();
            }
            
            // Label
            ctx.fillStyle = '#ffffff';
            ctx.font = '12px Inter';
            ctx.textAlign = 'center';
            ctx.fillText(agent.name, pos.x, pos.y + 40);
        });
    }

    showConsensusModal() {
        const modal = document.createElement('div');
        modal.className = 'agent-modal-overlay';
        modal.innerHTML = `
            <div class="agent-modal">
                <div class="agent-modal-header">
                    <h3>
                        <i class="fas fa-handshake"></i>
                        Consensus Analysis
                    </h3>
                    <button class="agent-modal-close">&times;</button>
                </div>
                <div class="agent-modal-content">
                    ${this.renderConsensusAnalysis()}
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        modal.querySelector('.agent-modal-close').addEventListener('click', () => {
            modal.remove();
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
    }

    renderConsensusAnalysis() {
        // Calculate agreement matrix
        const agents = Array.from(this.agents.values());
        
        return `
            <div class="consensus-grid">
                ${agents.map(agent => `
                    <div class="consensus-card">
                        <div class="consensus-agent" style="border-color: ${agent.color}">
                            <i class="${agent.icon}"></i>
                            ${agent.name}
                        </div>
                        <div class="consensus-meter">
                            <div class="consensus-fill" style="width: ${agent.confidence * 100}%; background: ${agent.color}"></div>
                        </div>
                        <div class="consensus-label">${(agent.confidence * 100).toFixed(0)}% confidence</div>
                    </div>
                `).join('')}
            </div>
            <div class="consensus-summary">
                <h4>Overall Consensus</h4>
                <p>The multi-agent system has reached ${this.calculateOverallConsensus()}% agreement on the current task.</p>
            </div>
        `;
    }

    calculateOverallConsensus() {
        const agents = Array.from(this.agents.values());
        const avg = agents.reduce((sum, agent) => sum + agent.confidence, 0) / agents.length;
        return (avg * 100).toFixed(0);
    }

    activateAgent(agentId) {
        this.updateAgentStatus(agentId, 'active', 'Taking control of task...');
        
        // Deactivate others
        this.agents.forEach((agent, id) => {
            if (id !== agentId && agent.status === 'active') {
                this.updateAgentStatus(id, 'idle');
            }
        });
    }

    viewAgentHistory(agentId) {
        console.log('View history for', agentId);
        // Implementation for showing agent's decision history
    }

    showAddAgentModal() {
        // Implementation for adding custom agents
        console.log('Add agent modal');
    }
}

// Global instance
window.multiAgentView = null;
