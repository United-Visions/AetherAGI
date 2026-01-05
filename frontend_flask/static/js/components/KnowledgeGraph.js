// components/KnowledgeGraph.js - 3D Interactive Knowledge Visualization

export class KnowledgeGraph {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.nodes = new Map();
        this.edges = [];
        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="knowledge-graph-container">
                <div class="knowledge-graph-controls">
                    <button class="kg-btn" id="kg-center" title="Center View">
                        <i class="fas fa-compress"></i>
                    </button>
                    <button class="kg-btn" id="kg-filter" title="Filter">
                        <i class="fas fa-filter"></i>
                    </button>
                    <button class="kg-btn" id="kg-search" title="Search">
                        <i class="fas fa-search"></i>
                    </button>
                    <div class="kg-legend">
                        <div class="kg-legend-item">
                            <div class="kg-dot" style="background: #10b981"></div>
                            <span>Your Knowledge</span>
                        </div>
                        <div class="kg-legend-item">
                            <div class="kg-dot" style="background: #3b82f6"></div>
                            <span>Core Knowledge</span>
                        </div>
                        <div class="kg-legend-item">
                            <div class="kg-dot" style="background: #f59e0b"></div>
                            <span>Recently Learned</span>
                        </div>
                    </div>
                </div>
                <canvas id="knowledge-graph-canvas"></canvas>
                <div class="knowledge-graph-sidebar">
                    <div class="kg-sidebar-content">
                        <h3>Knowledge Inspector</h3>
                        <div id="kg-node-details"></div>
                    </div>
                </div>
            </div>
        `;

        this.canvas = document.getElementById('knowledge-graph-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.sidebar = this.container.querySelector('.knowledge-graph-sidebar');
        this.detailsPanel = document.getElementById('kg-node-details');

        this.setupCanvas();
        this.setupControls();
        this.startAnimation();
    }

    setupCanvas() {
        const resize = () => {
            this.canvas.width = this.canvas.offsetWidth * window.devicePixelRatio;
            this.canvas.height = this.canvas.offsetHeight * window.devicePixelRatio;
            this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        };
        resize();
        window.addEventListener('resize', resize);

        // Mouse interaction
        let isDragging = false;
        let dragNode = null;

        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            dragNode = this.getNodeAt(x, y);
            if (dragNode) {
                isDragging = true;
                this.showNodeDetails(dragNode);
            }
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (isDragging && dragNode) {
                const rect = this.canvas.getBoundingClientRect();
                dragNode.x = e.clientX - rect.left;
                dragNode.y = e.clientY - rect.top;
            }
        });

        this.canvas.addEventListener('mouseup', () => {
            isDragging = false;
            dragNode = null;
        });

        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const node = this.getNodeAt(x, y);
            if (node) {
                this.showNodeDetails(node);
                this.sidebar.classList.add('active');
            }
        });
    }

    setupControls() {
        document.getElementById('kg-center').addEventListener('click', () => {
            this.centerView();
        });

        document.getElementById('kg-filter').addEventListener('click', () => {
            this.showFilterModal();
        });

        document.getElementById('kg-search').addEventListener('click', () => {
            this.showSearchModal();
        });
    }

    addNode(concept, metadata = {}) {
        const node = {
            id: concept,
            label: concept,
            x: Math.random() * this.canvas.width,
            y: Math.random() * this.canvas.height,
            vx: 0,
            vy: 0,
            radius: 8,
            color: metadata.color || '#3b82f6',
            type: metadata.type || 'core',
            learnedAt: metadata.learnedAt || new Date(),
            confidence: metadata.confidence || 0.8,
            connections: 0,
            metadata: metadata
        };

        this.nodes.set(concept, node);
        return node;
    }

    addEdge(from, to, relationship = 'relates_to') {
        const fromNode = this.nodes.get(from);
        const toNode = this.nodes.get(to);
        
        if (fromNode && toNode) {
            this.edges.push({ from: fromNode, to: toNode, relationship });
            fromNode.connections++;
            toNode.connections++;
        }
    }

    updateFromMemory(memoryData) {
        // Add nodes from memory retrieval
        if (memoryData.concepts) {
            memoryData.concepts.forEach(concept => {
                if (!this.nodes.has(concept.name)) {
                    this.addNode(concept.name, {
                        type: concept.namespace === 'user_episodic' ? 'user' : 'core',
                        color: concept.namespace === 'user_episodic' ? '#10b981' : '#3b82f6',
                        confidence: concept.confidence,
                        learnedAt: new Date(concept.timestamp)
                    });
                }
            });
        }

        // Add relationships
        if (memoryData.relationships) {
            memoryData.relationships.forEach(rel => {
                this.addEdge(rel.from, rel.to, rel.type);
            });
        }
    }

    getNodeAt(x, y) {
        for (const [, node] of this.nodes) {
            const dx = x - node.x;
            const dy = y - node.y;
            if (Math.sqrt(dx * dx + dy * dy) < node.radius) {
                return node;
            }
        }
        return null;
    }

    showNodeDetails(node) {
        const ageText = this.getAgeText(node.learnedAt);
        
        this.detailsPanel.innerHTML = `
            <div class="kg-node-card">
                <div class="kg-node-title">${node.label}</div>
                <div class="kg-node-meta">
                    <div class="kg-meta-item">
                        <i class="fas fa-link"></i>
                        <span>${node.connections} connections</span>
                    </div>
                    <div class="kg-meta-item">
                        <i class="fas fa-clock"></i>
                        <span>Learned ${ageText}</span>
                    </div>
                    <div class="kg-meta-item">
                        <i class="fas fa-chart-line"></i>
                        <span>${(node.confidence * 100).toFixed(0)}% confidence</span>
                    </div>
                </div>
                ${node.metadata.description ? `
                    <div class="kg-node-description">${node.metadata.description}</div>
                ` : ''}
                <div class="kg-node-actions">
                    <button class="kg-action-btn" onclick="kg.exploreNode('${node.id}')">
                        <i class="fas fa-project-diagram"></i>
                        Explore Connections
                    </button>
                    <button class="kg-action-btn" onclick="kg.forgetNode('${node.id}')">
                        <i class="fas fa-trash"></i>
                        Forget
                    </button>
                </div>
            </div>
        `;
    }

    startAnimation() {
        const animate = () => {
            this.update();
            this.render();
            requestAnimationFrame(animate);
        };
        animate();
    }

    update() {
        // Simple force-directed layout
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;

        // Apply forces
        for (const [, node] of this.nodes) {
            // Center attraction
            const dx = centerX - node.x;
            const dy = centerY - node.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            node.vx += (dx / dist) * 0.1;
            node.vy += (dy / dist) * 0.1;

            // Repulsion from other nodes
            for (const [, other] of this.nodes) {
                if (node === other) continue;
                const dx = node.x - other.x;
                const dy = node.y - other.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 100) {
                    const force = (100 - dist) / 100;
                    node.vx += (dx / dist) * force * 0.5;
                    node.vy += (dy / dist) * force * 0.5;
                }
            }

            // Apply velocity
            node.vx *= 0.9;
            node.vy *= 0.9;
            node.x += node.vx;
            node.y += node.vy;

            // Keep in bounds
            node.x = Math.max(node.radius, Math.min(this.canvas.width - node.radius, node.x));
            node.y = Math.max(node.radius, Math.min(this.canvas.height - node.radius, node.y));
        }
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw edges
        this.ctx.strokeStyle = 'rgba(100, 116, 139, 0.3)';
        this.ctx.lineWidth = 1;
        this.edges.forEach(edge => {
            this.ctx.beginPath();
            this.ctx.moveTo(edge.from.x, edge.from.y);
            this.ctx.lineTo(edge.to.x, edge.to.y);
            this.ctx.stroke();
        });

        // Draw nodes
        for (const [, node] of this.nodes) {
            // Glow effect for recent nodes
            const age = Date.now() - node.learnedAt;
            if (age < 5000) {
                this.ctx.shadowBlur = 20;
                this.ctx.shadowColor = node.color;
            } else {
                this.ctx.shadowBlur = 0;
            }

            this.ctx.fillStyle = node.color;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            this.ctx.fill();

            // Label
            this.ctx.shadowBlur = 0;
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '10px Inter';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(node.label, node.x, node.y - node.radius - 5);
        }
    }

    getAgeText(date) {
        const seconds = Math.floor((new Date() - date) / 1000);
        if (seconds < 60) return `${seconds}s ago`;
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        return `${days}d ago`;
    }

    centerView() {
        // Reset node positions to center
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        for (const [, node] of this.nodes) {
            node.x = centerX + (Math.random() - 0.5) * 200;
            node.y = centerY + (Math.random() - 0.5) * 200;
            node.vx = 0;
            node.vy = 0;
        }
    }

    showFilterModal() {
        // Implementation for filtering nodes
        console.log('Filter modal');
    }

    showSearchModal() {
        // Implementation for searching nodes
        console.log('Search modal');
    }
}

// Make it globally accessible for inline onclick handlers
window.kg = null;
