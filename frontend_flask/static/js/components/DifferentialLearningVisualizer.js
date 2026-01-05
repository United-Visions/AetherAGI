// components/DifferentialLearningVisualizer.js - Before/After Knowledge Comparison

export class DifferentialLearningVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.learningEvents = [];
        this.currentComparison = null;
        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="differential-container">
                <div class="differential-header">
                    <h3>
                        <i class="fas fa-brain"></i>
                        Differential Learning
                    </h3>
                    <div class="differential-stats">
                        <div class="stat-item">
                            <span class="stat-value" id="total-updates">0</span>
                            <span class="stat-label">Total Updates</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value" id="active-learning">0</span>
                            <span class="stat-label">Active Learning</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value" id="promoted-memories">0</span>
                            <span class="stat-label">Promoted</span>
                        </div>
                    </div>
                </div>

                <div class="differential-content">
                    <!-- Comparison View -->
                    <div class="comparison-section">
                        <div class="comparison-panel before-panel">
                            <div class="panel-header">
                                <i class="fas fa-history"></i>
                                <h4>Before Learning</h4>
                            </div>
                            <div class="knowledge-display" id="knowledge-before">
                                <div class="knowledge-empty">
                                    <i class="fas fa-brain"></i>
                                    <p>No previous knowledge state</p>
                                </div>
                            </div>
                        </div>

                        <div class="comparison-divider">
                            <div class="learning-arrow">
                                <i class="fas fa-arrow-right"></i>
                                <span>Learning Applied</span>
                            </div>
                        </div>

                        <div class="comparison-panel after-panel">
                            <div class="panel-header">
                                <i class="fas fa-sparkles"></i>
                                <h4>After Learning</h4>
                            </div>
                            <div class="knowledge-display" id="knowledge-after">
                                <div class="knowledge-empty">
                                    <i class="fas fa-brain"></i>
                                    <p>No updated knowledge state</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Delta Visualization -->
                    <div class="delta-section">
                        <h4>
                            <i class="fas fa-chart-line"></i>
                            Knowledge Delta
                        </h4>
                        <div class="delta-metrics" id="delta-metrics"></div>
                    </div>

                    <!-- Memory Promotion Tracker -->
                    <div class="promotion-section">
                        <h4>
                            <i class="fas fa-arrow-up"></i>
                            Memory Promotion Pipeline
                        </h4>
                        <div class="promotion-pipeline" id="promotion-pipeline">
                            <div class="pipeline-stage">
                                <div class="stage-header">Episodic</div>
                                <div class="stage-content" id="stage-episodic">
                                    <span class="stage-count">0</span>
                                    <span class="stage-label">memories</span>
                                </div>
                            </div>
                            <div class="pipeline-arrow">
                                <i class="fas fa-arrow-right"></i>
                            </div>
                            <div class="pipeline-stage">
                                <div class="stage-header">Semantic</div>
                                <div class="stage-content" id="stage-semantic">
                                    <span class="stage-count">0</span>
                                    <span class="stage-label">concepts</span>
                                </div>
                            </div>
                            <div class="pipeline-arrow">
                                <i class="fas fa-arrow-right"></i>
                            </div>
                            <div class="pipeline-stage">
                                <div class="stage-header">Core</div>
                                <div class="stage-content" id="stage-core">
                                    <span class="stage-count">0</span>
                                    <span class="stage-label">priors</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Gradient Flow Animation -->
                    <div class="gradient-section">
                        <h4>
                            <i class="fas fa-wave-square"></i>
                            Learning Gradient Flow
                        </h4>
                        <canvas id="gradient-canvas"></canvas>
                        <div class="gradient-legend">
                            <div class="legend-item">
                                <div class="gradient-dot positive"></div>
                                <span>Positive Update</span>
                            </div>
                            <div class="legend-item">
                                <div class="gradient-dot negative"></div>
                                <span>Correction</span>
                            </div>
                            <div class="legend-item">
                                <div class="gradient-dot neutral"></div>
                                <span>Refinement</span>
                            </div>
                        </div>
                    </div>

                    <!-- Learning History Timeline -->
                    <div class="history-section">
                        <h4>
                            <i class="fas fa-timeline"></i>
                            Learning History
                        </h4>
                        <div class="learning-timeline" id="learning-timeline"></div>
                    </div>
                </div>
            </div>
        `;

        this.setupCanvas();
        this.startGradientAnimation();
    }

    setupCanvas() {
        this.canvas = document.getElementById('gradient-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        const resize = () => {
            this.canvas.width = this.canvas.offsetWidth * window.devicePixelRatio;
            this.canvas.height = 200 * window.devicePixelRatio;
            this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        };
        resize();
        window.addEventListener('resize', resize);
        
        this.gradients = [];
    }

    addLearningEvent(event) {
        /*
        event structure:
        {
            id: unique_id,
            type: 'new' | 'correction' | 'refinement',
            before: { knowledge object },
            after: { knowledge object },
            delta: { changes },
            promoted: boolean,
            timestamp: Date
        }
        */
        this.learningEvents.unshift(event);
        
        // Update stats
        document.getElementById('total-updates').textContent = this.learningEvents.length;
        
        if (event.promoted) {
            const promoted = document.getElementById('promoted-memories');
            promoted.textContent = parseInt(promoted.textContent) + 1;
        }
        
        // Add to timeline
        this.addToTimeline(event);
        
        // Show in comparison view
        this.showComparison(event);
        
        // Add gradient flow
        this.addGradient(event);
        
        // Update promotion pipeline
        this.updatePromotionPipeline();
    }

    showComparison(event) {
        this.currentComparison = event;
        
        const beforePanel = document.getElementById('knowledge-before');
        const afterPanel = document.getElementById('knowledge-after');
        
        // Render before state
        beforePanel.innerHTML = this.renderKnowledgeState(event.before);
        
        // Render after state with highlights
        afterPanel.innerHTML = this.renderKnowledgeState(event.after, event.delta);
        
        // Animate transition
        afterPanel.classList.add('learning-glow');
        setTimeout(() => afterPanel.classList.remove('learning-glow'), 1000);
        
        // Render delta metrics
        this.renderDeltaMetrics(event.delta);
    }

    renderKnowledgeState(knowledge, delta = null) {
        if (!knowledge || Object.keys(knowledge).length === 0) {
            return `
                <div class="knowledge-empty">
                    <i class="fas fa-brain"></i>
                    <p>No knowledge state</p>
                </div>
            `;
        }
        
        let html = '<div class="knowledge-tree">';
        
        for (const [key, value] of Object.entries(knowledge)) {
            const isChanged = delta && delta.hasOwnProperty(key);
            const changeType = isChanged ? this.getChangeType(value, delta[key]) : null;
            
            html += `
                <div class="knowledge-node ${isChanged ? 'changed' : ''} ${changeType || ''}">
                    <div class="node-key">
                        ${isChanged ? '<i class="fas fa-bolt"></i>' : '<i class="fas fa-circle"></i>'}
                        ${key}
                    </div>
                    <div class="node-value">
                        ${this.formatValue(value)}
                        ${isChanged ? `<span class="change-badge">${changeType}</span>` : ''}
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        return html;
    }

    getChangeType(oldValue, newValue) {
        if (oldValue === undefined || oldValue === null) return 'added';
        if (newValue === undefined || newValue === null) return 'removed';
        
        // Compare confidence/certainty if available
        if (typeof oldValue === 'object' && oldValue.confidence !== undefined) {
            if (newValue.confidence > oldValue.confidence) return 'strengthened';
            if (newValue.confidence < oldValue.confidence) return 'weakened';
        }
        
        return 'modified';
    }

    formatValue(value) {
        if (typeof value === 'object') {
            if (value.confidence !== undefined) {
                return `
                    <div class="value-with-confidence">
                        <span class="value-text">${value.value || JSON.stringify(value)}</span>
                        <div class="confidence-mini-bar">
                            <div class="confidence-mini-fill" style="width: ${value.confidence * 100}%"></div>
                        </div>
                        <span class="confidence-value">${(value.confidence * 100).toFixed(0)}%</span>
                    </div>
                `;
            }
            return JSON.stringify(value, null, 2);
        }
        return value;
    }

    renderDeltaMetrics(delta) {
        const metricsContainer = document.getElementById('delta-metrics');
        
        if (!delta || Object.keys(delta).length === 0) {
            metricsContainer.innerHTML = '<p class="delta-empty">No changes detected</p>';
            return;
        }
        
        const changes = {
            added: 0,
            removed: 0,
            modified: 0,
            strengthened: 0,
            weakened: 0
        };
        
        for (const [key, value] of Object.entries(delta)) {
            const changeType = this.getChangeType(
                this.currentComparison?.before?.[key],
                value
            );
            if (changes.hasOwnProperty(changeType)) {
                changes[changeType]++;
            } else {
                changes.modified++;
            }
        }
        
        metricsContainer.innerHTML = `
            <div class="delta-cards">
                ${changes.added > 0 ? `
                    <div class="delta-card added">
                        <i class="fas fa-plus-circle"></i>
                        <span class="delta-count">${changes.added}</span>
                        <span class="delta-label">Added</span>
                    </div>
                ` : ''}
                ${changes.removed > 0 ? `
                    <div class="delta-card removed">
                        <i class="fas fa-minus-circle"></i>
                        <span class="delta-count">${changes.removed}</span>
                        <span class="delta-label">Removed</span>
                    </div>
                ` : ''}
                ${changes.modified > 0 ? `
                    <div class="delta-card modified">
                        <i class="fas fa-edit"></i>
                        <span class="delta-count">${changes.modified}</span>
                        <span class="delta-label">Modified</span>
                    </div>
                ` : ''}
                ${changes.strengthened > 0 ? `
                    <div class="delta-card strengthened">
                        <i class="fas fa-arrow-up"></i>
                        <span class="delta-count">${changes.strengthened}</span>
                        <span class="delta-label">Strengthened</span>
                    </div>
                ` : ''}
                ${changes.weakened > 0 ? `
                    <div class="delta-card weakened">
                        <i class="fas fa-arrow-down"></i>
                        <span class="delta-count">${changes.weakened}</span>
                        <span class="delta-label">Weakened</span>
                    </div>
                ` : ''}
            </div>
        `;
    }

    updatePromotionPipeline() {
        // Count memories in each stage
        const stages = {
            episodic: 0,
            semantic: 0,
            core: 0
        };
        
        this.learningEvents.forEach(event => {
            if (event.stage) {
                stages[event.stage]++;
            } else {
                stages.episodic++; // Default
            }
        });
        
        document.querySelector('#stage-episodic .stage-count').textContent = stages.episodic;
        document.querySelector('#stage-semantic .stage-count').textContent = stages.semantic;
        document.querySelector('#stage-core .stage-count').textContent = stages.core;
    }

    addToTimeline(event) {
        const timeline = document.getElementById('learning-timeline');
        
        const typeIcons = {
            'new': 'fa-plus',
            'correction': 'fa-wrench',
            'refinement': 'fa-sliders'
        };
        
        const typeColors = {
            'new': '#10b981',
            'correction': '#ef4444',
            'refinement': '#3b82f6'
        };
        
        const item = document.createElement('div');
        item.className = 'timeline-item fade-in';
        item.innerHTML = `
            <div class="timeline-marker" style="background: ${typeColors[event.type] || '#6b7280'}">
                <i class="fas ${typeIcons[event.type] || 'fa-circle'}"></i>
            </div>
            <div class="timeline-content">
                <div class="timeline-header">
                    <span class="timeline-type">${event.type}</span>
                    <span class="timeline-time">${new Date(event.timestamp).toLocaleTimeString()}</span>
                </div>
                <div class="timeline-description">
                    ${Object.keys(event.delta || {}).length} knowledge nodes updated
                    ${event.promoted ? '<span class="promoted-badge"><i class="fas fa-star"></i> Promoted</span>' : ''}
                </div>
                <button class="timeline-view-btn" onclick="differentialVisualizer.showComparison(${JSON.stringify(event).replace(/"/g, '&quot;')})">
                    View Details
                </button>
            </div>
        `;
        
        timeline.insertBefore(item, timeline.firstChild);
        
        // Keep only last 20
        while (timeline.children.length > 20) {
            timeline.removeChild(timeline.lastChild);
        }
    }

    addGradient(event) {
        const type = event.type;
        const color = {
            'new': { r: 16, g: 185, b: 129 },      // Green
            'correction': { r: 239, g: 68, b: 68 }, // Red
            'refinement': { r: 59, g: 130, b: 246 } // Blue
        }[type] || { r: 107, g: 114, b: 128 };     // Gray
        
        this.gradients.push({
            x: 0,
            y: Math.random() * (this.canvas.offsetHeight / window.devicePixelRatio),
            speed: 2 + Math.random() * 3,
            amplitude: 20 + Math.random() * 30,
            frequency: 0.02 + Math.random() * 0.03,
            color: color,
            opacity: 1,
            phase: Math.random() * Math.PI * 2
        });
    }

    startGradientAnimation() {
        const animate = () => {
            this.renderGradients();
            requestAnimationFrame(animate);
        };
        animate();
    }

    renderGradients() {
        const ctx = this.ctx;
        const width = this.canvas.offsetWidth;
        const height = this.canvas.offsetHeight / window.devicePixelRatio;
        
        ctx.clearRect(0, 0, width, height);
        
        // Update and draw gradients
        this.gradients = this.gradients.filter(gradient => {
            gradient.x += gradient.speed;
            gradient.opacity -= 0.002;
            
            if (gradient.x > width || gradient.opacity <= 0) {
                return false;
            }
            
            // Draw wave
            ctx.strokeStyle = `rgba(${gradient.color.r}, ${gradient.color.g}, ${gradient.color.b}, ${gradient.opacity})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let i = 0; i < width - gradient.x; i += 2) {
                const x = gradient.x + i;
                const y = gradient.y + Math.sin((i * gradient.frequency) + gradient.phase) * gradient.amplitude;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            
            ctx.stroke();
            
            return true;
        });
    }

    // Public API for external updates
    recordLearning(before, after, metadata = {}) {
        const delta = this.calculateDelta(before, after);
        
        const event = {
            id: Date.now(),
            type: metadata.type || 'refinement',
            before: before,
            after: after,
            delta: delta,
            promoted: metadata.promoted || false,
            stage: metadata.stage || 'episodic',
            timestamp: new Date()
        };
        
        this.addLearningEvent(event);
    }

    calculateDelta(before, after) {
        const delta = {};
        
        // Find added/modified
        for (const key in after) {
            if (JSON.stringify(before[key]) !== JSON.stringify(after[key])) {
                delta[key] = after[key];
            }
        }
        
        // Find removed
        for (const key in before) {
            if (!(key in after)) {
                delta[key] = null;
            }
        }
        
        return delta;
    }
}

// Global instance
window.differentialVisualizer = null;
