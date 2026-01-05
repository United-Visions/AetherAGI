// components/BrainVisualizer.js - Visual representation of Agent's thinking process

export class BrainVisualizer {
    constructor(containerId) {
        console.log('🧠 [BrainVisualizer] Constructor called with containerId:', containerId);
        this.container = document.getElementById(containerId);
        
        if (!this.container) {
            console.error('❌ [BrainVisualizer] Container not found:', containerId);
            return;
        }
        
        this.nodes = [];
        this.connections = [];
        this.animationFrame = null;
        console.log('✅ [BrainVisualizer] Properties initialized');
        
        this.init();
    }

    init() {
        console.log('🚀 [BrainVisualizer] Initializing brain visualizer UI...');
        this.container.innerHTML = `
            <div class="brain-visualizer">
                <div class="brain-header">
                    <i class="fas fa-brain"></i>
                    <span>Active Inference Loop</span>
                    <div class="brain-status">
                        <span class="status-dot"></span>
                        <span id="brain-status-text">Idle</span>
                    </div>
                </div>
                
                <div class="brain-canvas-wrapper">
                    <canvas id="brain-canvas" width="400" height="300"></canvas>
                    
                    <div class="brain-stages" id="brain-stages">
                        <!-- Stages populated dynamically -->
                    </div>
                </div>
                
                <div class="brain-metrics" id="brain-metrics">
                    <div class="metric">
                        <span class="metric-label">Surprise Score</span>
                        <span class="metric-value" id="surprise-score">0.00</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Confidence</span>
                        <span class="metric-value" id="confidence-score">0.00</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Processing</span>
                        <span class="metric-value" id="processing-time">0ms</span>
                    </div>
                </div>
            </div>
        `;

        this.canvas = document.getElementById('brain-canvas');
        this.ctx = this.canvas.getContext('2d');
        console.log('✅ [BrainVisualizer] Canvas initialized');
        
        this.setupStages();
        console.log('✅ [BrainVisualizer] UI initialization complete');
    }

    setupStages() {
        console.log('⚙️ [BrainVisualizer] Setting up stages...');
        const stages = [
            { id: 'sense', name: 'Sense', icon: 'fas fa-eye', color: '#10b981' },
            { id: 'retrieve', name: 'Retrieve', icon: 'fas fa-database', color: '#3b82f6' },
            { id: 'reason', name: 'Reason', icon: 'fas fa-brain', color: '#8b5cf6' },
            { id: 'embellish', name: 'Embellish', icon: 'fas fa-heart', color: '#ec4899' },
            { id: 'act', name: 'Act', icon: 'fas fa-bolt', color: '#f59e0b' },
            { id: 'learn', name: 'Learn', icon: 'fas fa-graduation-cap', color: '#06b6d4' }
        ];

        const stagesContainer = document.getElementById('brain-stages');
        stagesContainer.innerHTML = stages.map(stage => `
            <div class="brain-stage" id="stage-${stage.id}" data-stage="${stage.id}">
                <div class="stage-icon" style="background: ${stage.color}20; color: ${stage.color};">
                    <i class="${stage.icon}"></i>
                </div>
                <div class="stage-name">${stage.name}</div>
                <div class="stage-indicator"></div>
            </div>
        `).join('');
    }

    startThinking(metadata = {}) {
        document.getElementById('brain-status-text').textContent = 'Processing';
        this.setStageActive('sense');
        
        // Simulate thinking stages
        const stages = ['sense', 'retrieve', 'reason', 'embellish', 'act', 'learn'];
        stages.forEach((stage, index) => {
            setTimeout(() => {
                this.setStageActive(stage);
                if (index === stages.length - 1) {
                    setTimeout(() => {
                        this.stopThinking();
                    }, 500);
                }
            }, index * 300);
        });

        // Update metrics
        if (metadata.surprise_score !== undefined) {
            document.getElementById('surprise-score').textContent = metadata.surprise_score.toFixed(2);
        }
        if (metadata.confidence !== undefined) {
            document.getElementById('confidence-score').textContent = metadata.confidence.toFixed(2);
        }
    }

    stopThinking() {
        document.getElementById('brain-status-text').textContent = 'Idle';
        this.clearActiveStages();
    }

    updateMetrics(metrics = {}) {
        console.log('📊 [BrainVisualizer] Updating metrics:', metrics);
        
        // Update surprise score
        if (metrics.surprise_score !== undefined) {
            const surpriseEl = document.getElementById('surprise-score');
            if (surpriseEl) {
                surpriseEl.textContent = metrics.surprise_score.toFixed(2);
                // Highlight if high surprise
                if (metrics.surprise_score > 0.5) {
                    surpriseEl.style.color = '#f59e0b';
                } else {
                    surpriseEl.style.color = '#10b981';
                }
            }
        }
        
        // Update confidence
        if (metrics.confidence !== undefined) {
            const confidenceEl = document.getElementById('confidence-score');
            if (confidenceEl) {
                confidenceEl.textContent = (metrics.confidence * 100).toFixed(0) + '%';
                // Highlight if low confidence
                if (metrics.confidence < 0.5) {
                    confidenceEl.style.color = '#ef4444';
                } else {
                    confidenceEl.style.color = '#10b981';
                }
            }
        }
        
        // Update processing time
        if (metrics.response_time !== undefined) {
            const processingEl = document.getElementById('processing-time');
            if (processingEl) {
                processingEl.textContent = Math.round(metrics.response_time) + 'ms';
            }
        }
        
        console.log('✅ [BrainVisualizer] Metrics updated successfully');
    }

    setStageActive(stageId) {
        // Clear previous active
        document.querySelectorAll('.brain-stage').forEach(stage => {
            stage.classList.remove('active', 'completed');
        });

        // Set current and previous as completed
        const stages = ['sense', 'retrieve', 'reason', 'embellish', 'act', 'learn'];
        const currentIndex = stages.indexOf(stageId);
        
        stages.forEach((id, index) => {
            const stageEl = document.getElementById(`stage-${id}`);
            if (index < currentIndex) {
                stageEl.classList.add('completed');
            } else if (index === currentIndex) {
                stageEl.classList.add('active');
            }
        });
    }

    clearActiveStages() {
        document.querySelectorAll('.brain-stage').forEach(stage => {
            stage.classList.remove('active');
            stage.classList.add('completed');
        });
        
        setTimeout(() => {
            document.querySelectorAll('.brain-stage').forEach(stage => {
                stage.classList.remove('completed');
            });
        }, 1000);
    }

    // Canvas animation for neural network visualization
    drawNeuralNetwork() {
        if (!this.ctx) return;
        
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.clearRect(0, 0, width, height);
        
        // Draw connections (simplified)
        this.ctx.strokeStyle = 'rgba(16, 185, 129, 0.2)';
        this.ctx.lineWidth = 1;
        
        for (let i = 0; i < 20; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo(Math.random() * width, Math.random() * height);
            this.ctx.lineTo(Math.random() * width, Math.random() * height);
            this.ctx.stroke();
        }
        
        // Draw nodes
        this.ctx.fillStyle = 'rgba(16, 185, 129, 0.5)';
        for (let i = 0; i < 30; i++) {
            this.ctx.beginPath();
            this.ctx.arc(
                Math.random() * width,
                Math.random() * height,
                3,
                0,
                Math.PI * 2
            );
            this.ctx.fill();
        }
    }
}
