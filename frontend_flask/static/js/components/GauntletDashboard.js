/**
 * GauntletDashboard.js
 * Clean version of the benchmark execution and visualization system.
 * Manages AetherMind performance gauntlets and live progress.
 */

class GauntletDashboard {
    constructor(sidebar) {
        this.sidebar = sidebar;
        this.backendUrl = sidebar.backendUrl;
        this.activeRuns = new Map();
        this.pollInterval = null;
        this.selectedBenchmarks = new Set();
        this.benchmarksData = [];
        console.log('🛡️ [GAUNTLET] Dashboard initialized');
    }

    getApiKey() {
        // First check URL params and store if found
        const urlParams = new URLSearchParams(window.location.search);
        const urlKey = urlParams.get('api_key');
        if (urlKey) {
            localStorage.setItem('aethermind_api_key', urlKey);
            return urlKey;
        }

        const canonicalKey = localStorage.getItem('aethermind_api_key');
        if (canonicalKey) return canonicalKey;

        const legacyKey = localStorage.getItem('aether_api_key');
        if (legacyKey) {
            // Migrate to canonical name
            localStorage.setItem('aethermind_api_key', legacyKey);
            return legacyKey;
        }
        
        return null;
    }

    async loadBenchmarks() {
        const benchmarksList = document.getElementById('benchmarks-list');
        if (!benchmarksList) return;

        try {
            const apiKey = this.getApiKey();
            if (!apiKey) {
                benchmarksList.innerHTML = '<div class="tool-item error-state" style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> No API Key found. Please log in.</div>';
                return;
            }
            
            console.log('🛡️ [GAUNTLET] Loading benchmarks with API key:', apiKey.substring(0, 15) + '...');
            
            const response = await fetch(this.backendUrl + '/v1/benchmarks', {
                headers: { 'X-Aether-Key': apiKey }
            });

            if (response.ok) {
                this.benchmarksData = await response.json();
                this.renderBenchmarks();
                this.renderSummary();
            } else if (response.status === 403) {
                benchmarksList.innerHTML = '<div class="tool-item error-state" style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> Access Denied (403). Check API Key.</div>';
            }
        } catch (error) {
            console.error('🛡️ [GAUNTLET] Error loading benchmarks:', error);
        }
    }

    renderBenchmarks() {
        const benchmarksList = document.getElementById('benchmarks-list');
        if (!benchmarksList) return;

        benchmarksList.innerHTML = this.benchmarksData.map(b => {
            const score = b.last_results ? Math.round(b.last_results.overall_average * 100) : null;
            const isSelected = this.selectedBenchmarks.has(b.id);
            return `
                <div class="tool-item benchmark-item ${isSelected ? 'selected' : ''}" 
                     data-id="${b.id}" 
                     onclick="window.gauntletDashboard.toggleSelection('${b.id}')">
                    <i class="fas fa-flask" style="color: ${isSelected ? '#10b981' : '#3b82f6'};"></i>
                    <div class="tool-item-info">
                        <div class="tool-item-name">${b.name}</div>
                        <div class="tool-item-desc">${b.description}</div>
                        ${score !== null ? `<div class="tool-item-desc" style="color: #10b981; font-weight: 600;">Last Score: ${score}%</div>` : ''}
                    </div>
                    <button class="run-benchmark-btn" onclick="event.stopPropagation(); window.gauntletDashboard.showModelSelector('${b.id}')">
                        <i class="fas fa-play"></i>
                    </button>
                </div>
            `;
        }).join('');
    }

    renderSummary() {
        const benchmarkView = document.getElementById('view-benchmarks');
        if (!benchmarkView) return;

        let controls = benchmarkView.querySelector('.benchmark-controls');
        if (!controls) {
            controls = document.createElement('div');
            controls.className = 'benchmark-controls';
            benchmarkView.prepend(controls);
        }
        
        controls.style.cssText = 'margin-bottom: 1.5rem; background: var(--bg-tertiary); border-radius: 12px; padding: 15px; border: 1px solid var(--border-color);';

        const totalAverage = this.benchmarksData.length > 0
            ? this.benchmarksData.reduce((acc, b) => {
                return acc + (b.last_results ? b.last_results.overall_average : 0);
            }, 0) / this.benchmarksData.length
            : 0;

        controls.innerHTML = `
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.85em; font-weight: 600; color: var(--text-muted);">AVERAGE GAUNTLET SCORE</span>
                    <span style="font-size: 1.1em; font-weight: 700; color: #10b981;">${Math.round(totalAverage * 100)}%</span>
                </div>
                <div class="benchmark-progress-bar" style="height: 8px; border-radius: 4px; background: var(--bg-secondary); overflow: hidden;">
                    <div class="benchmark-progress-fill" style="width: ${totalAverage * 100}%; background: linear-gradient(90deg, #10b981, #3b82f6); height: 100%; border-radius: 4px; transition: width 1s ease-out;"></div>
                </div>
            </div>

            <div style="display: flex; flex-direction: column; gap: 8px;">
                <button class="sidebar-add-btn" onclick="window.gauntletDashboard.activateAetherGauntlet()" style="margin: 0; background: linear-gradient(135deg, #10b981 0%, #059669 100%); border: none; height: 40px; font-size: 12px; font-weight: 700;">
                    <i class="fas fa-bolt"></i> ACTIVATE AETHER GAUNTLET
                </button>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <button class="sidebar-add-btn" onclick="window.gauntletDashboard.showPerformanceView()" style="margin: 0; background: var(--bg-secondary); border: 1px solid var(--border-color); height: 36px; font-size: 10px;">
                        <i class="fas fa-chart-line"></i> REAL VIEW
                    </button>
                    <button class="sidebar-add-btn" onclick="window.gauntletDashboard.testBigThreeTheory()" style="margin: 0; background: var(--bg-secondary); border: 1px solid var(--border-color); height: 36px; font-size: 10px;">
                        <i class="fas fa-microchip"></i> BIG THREE
                    </button>
                </div>
            </div>
        `;

        // Hide old redundant buttons
        const oldBtns = document.getElementById('activate-aether-btn')?.parentElement;
        if (oldBtns) oldBtns.style.display = 'none';
    }

    toggleSelection(id) {
        if (this.selectedBenchmarks.has(id)) {
            this.selectedBenchmarks.delete(id);
        } else {
            this.selectedBenchmarks.add(id);
        }
        this.renderBenchmarks();
    }

    showModelSelector(family) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 500px;">
                <div class="modal-header">
                    <h3 class="modal-title">Select Model for ${family.toUpperCase()}</h3>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()"><i class="fas fa-times"></i></button>
                </div>
                <div class="modal-body">
                    <div class="model-group">
                        <h4 style="margin: 10px 0; font-size: 0.9em; color: var(--text-muted); text-transform: uppercase;">Google Gemini</h4>
                        <div class="model-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <button class="model-btn" onclick="window.gauntletDashboard.startRun('${family}', 'gemini/gemini-2.5-pro')">Gemini 2.5 Pro</button>
                            <button class="model-btn" onclick="window.gauntletDashboard.startRun('${family}', 'gemini/gemini-2.5-flash')">Gemini 2.5 Flash</button>
                            <button class="model-btn" onclick="window.gauntletDashboard.startRun('${family}', 'gemini/gemini-3.0-pro-preview')">Gemini 3.0 Pro</button>
                            <button class="model-btn" onclick="window.gauntletDashboard.startRun('${family}', 'gemini/gemini-3.0-flash-preview')">Gemini 3.0 Flash</button>
                        </div>
                    </div>
                    
                    <div class="model-group" style="margin-top: 20px;">
                        <h4 style="margin: 10px 0; font-size: 0.9em; color: var(--text-muted); text-transform: uppercase;">OpenAI</h4>
                        <div class="model-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <button class="model-btn" onclick="window.gauntletDashboard.startRun('${family}', 'openai/gpt-4o')">GPT-4o</button>
                            <button class="model-btn" onclick="window.gauntletDashboard.startRun('${family}', 'openai/o1-preview')">OpenAI o1</button>
                        </div>
                    </div>

                    <div class="model-group" style="margin-top: 20px;">
                        <h4 style="margin: 10px 0; font-size: 0.9em; color: var(--text-muted); text-transform: uppercase;">Anthropic</h4>
                        <div class="model-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <button class="model-btn" onclick="window.gauntletDashboard.startRun('${family}', 'anthropic/claude-3-5-sonnet')">Claude 3.5 Sonnet</button>
                            <button class="model-btn" onclick="window.gauntletDashboard.startRun('${family}', 'anthropic/claude-3-opus')">Claude 3 Opus</button>
                        </div>
                    </div>
                    
                    <button class="modal-btn modal-btn-primary" style="margin-top: 20px; width: 100%;" onclick="window.gauntletDashboard.startRun('${family}', null)">
                        <i class="fas fa-robot"></i> Run with Aether (All-In)
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    async startRun(family, model) {
        try {
            const apiKey = this.getApiKey();
            const response = await fetch(`${this.backendUrl}/v1/benchmarks/run?family=${family}${model ? `&model=${model}` : ''}`, {
                method: 'POST',
                headers: { 'X-Aether-Key': apiKey }
            });

            if (response.ok) {
                const data = await response.json();
                this.activeRuns.set(data.benchmark_id, { family, model: model || 'Aether', status: 'started', progress: 0 });
                
                const modal = document.querySelector('.modal-overlay');
                if (modal) modal.remove();
                
                this.showProgressModal(data.benchmark_id);
                this.updateActiveRunsUI();
                this.startPolling();
                
                if (window.activityFeed) {
                    window.activityFeed.addActivity({
                        id: `bench_${data.benchmark_id}`,
                        type: 'benchmark',
                        status: 'in_progress',
                        title: `Benchmark Started: ${family.toUpperCase()}`,
                        details: `Running on ${model || 'AetherMind'}...`,
                        timestamp: new Date().toISOString(),
                        data: { family, model }
                    });
                }
            }
        } catch (error) {
            console.error('🛡️ [GAUNTLET] Error starting benchmark:', error);
        }
    }

    showProgressModal(id) {
        if (document.getElementById('benchmark-progress-modal')) return;

        const run = this.activeRuns.get(id);
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.id = 'benchmark-progress-modal';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 600px;">
                <div class="modal-header">
                    <h3 class="modal-title">Live Gauntlet Progress</h3>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()"><i class="fas fa-times"></i></button>
                </div>
                <div class="modal-body">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <h2 style="margin: 0;">${run.family.toUpperCase()}</h2>
                        <span style="color: var(--text-muted); text-transform: none;">${run.model}</span>
                    </div>
                    <div class="benchmark-progress-container" style="margin-bottom: 20px;">
                        <div class="benchmark-progress-bar" style="height: 12px; border-radius: 6px;">
                            <div id="modal-progress-fill" class="benchmark-progress-fill" style="width: 0%;"></div>
                        </div>
                        <div class="benchmark-status" style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.8em; font-weight: 600;">
                            <span id="modal-status-text">INITIALIZING...</span>
                            <span id="modal-progress-text">0%</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    async activateAetherGauntlet() {
        const benchmarks = this.selectedBenchmarks.size > 0 
            ? Array.from(this.selectedBenchmarks) 
            : ['gsm', 'mmlu', 'humaneval'];
            
        for (const family of benchmarks) {
            await this.startRun(family, null);
            await new Promise(r => setTimeout(r, 500));
        }
    }

    updateActiveRunsUI() {
        const activeList = document.getElementById('active-benchmarks-list');
        if (!activeList) return;

        if (this.activeRuns.size === 0) {
            activeList.innerHTML = `
                <div class="tool-item empty-state" style="border: 1px dashed var(--border-color); cursor: default;">
                    <i class="fas fa-microchip" style="color: var(--text-muted);"></i>
                    <div class="tool-item-info">
                        <div class="tool-item-desc">No active benchmark runs</div>
                    </div>
                </div>
            `;
            return;
        }

        activeList.innerHTML = Array.from(this.activeRuns.entries()).map(([id, run]) => {
            const progress = run.progress || 0;
            const statusColor = run.status === 'completed' ? '#10b981' : run.status === 'failed' ? '#ef4444' : '#3b82f6';
            
            return `
                <div class="tool-item" style="flex-direction: column; align-items: stretch; gap: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div class="tool-item-info">
                            <div class="tool-item-name">${run.family.toUpperCase()} (${run.model})</div>
                            <div class="tool-item-desc">${run.status.toUpperCase()}</div>
                        </div>
                        <div class="tool-item-badge" style="background: ${statusColor}20; color: ${statusColor}; border-color: ${statusColor}40;">
                            ${Math.round(progress)}%
                        </div>
                    </div>
                    <div style="height: 4px; background: var(--border-color); border-radius: 2px; overflow: hidden;">
                        <div style="height: 100%; width: ${progress}%; background: ${statusColor}; transition: width 0.3s ease;"></div>
                    </div>
                </div>
            `;
        }).join('');
    }

    startPolling() {
        if (this.pollInterval) return;
        
        this.pollInterval = setInterval(async () => {
            const apiKey = this.getApiKey();
            let hasActive = false;
            
            for (const [id, run] of this.activeRuns.entries()) {
                if (run.status === 'completed' || run.status === 'failed') continue;
                
                hasActive = true;
                try {
                    const response = await fetch(`${this.backendUrl}/v1/benchmarks/${id}/status`, {
                        headers: { 'X-Aether-Key': apiKey }
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        this.activeRuns.set(id, { ...run, ...data });
                        
                        const modal = document.getElementById('benchmark-progress-modal');
                        if (modal) {
                            const fill = document.getElementById('modal-progress-fill');
                            const text = document.getElementById('modal-progress-text');
                            const status = document.getElementById('modal-status-text');
                            
                            if (fill) fill.style.width = `${data.progress}%`;
                            if (text) text.textContent = `${Math.round(data.progress)}%`;
                            if (status) status.textContent = data.status.toUpperCase();
                        }
                    }
                } catch (e) {}
            }
            
            this.updateActiveRunsUI();
            
            if (!hasActive) {
                clearInterval(this.pollInterval);
                this.pollInterval = null;
                this.loadBenchmarks();
                this.showPerformanceView();
            }
        }, 3000);
    }

    async showPerformanceView() {
        const container = document.getElementById('split-view-container');
        if (!container) return;

        const apiKey = this.getApiKey();
        const response = await fetch(this.backendUrl + '/v1/benchmarks', {
            headers: { 'X-Aether-Key': apiKey }
        });

        if (!response.ok) return;
        const benchmarks = await response.json();
        
        const totalAverage = benchmarks.length > 0
            ? benchmarks.reduce((acc, b) => acc + (b.last_results ? b.last_results.overall_average : 0), 0) / benchmarks.length
            : 0;

        const content = `
            <div class="benchmark-full-report" style="padding: 2.5rem; color: var(--text-primary); animation: fadeIn 0.4s ease-out;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 3rem;">
                    <div>
                        <h1 style="margin: 0; font-size: 2.5em; letter-spacing: -0.5px;">Performance <span style="background: linear-gradient(90deg, #10b981, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Gauntlet</span></h1>
                        <p style="color: var(--text-muted); font-size: 1.1em; margin-top: 5px;">Agentic monitoring across ${benchmarks.length} logic & code dimensions.</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="display: flex; gap: 20px;">
                            <div class="stat-card">
                                <div style="font-size: 0.7em; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Logic Density</div>
                                <div style="font-size: 1.4em; font-weight: 700; color: #3b82f6;">9.4/10</div>
                            </div>
                            <div class="stat-card">
                                <div style="font-size: 0.7em; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Refinement</div>
                                <div style="font-size: 1.4em; font-weight: 700; color: #10b981;">OPTIMAL</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Master Slider -->
                <div style="background: var(--bg-secondary); padding: 30px; border-radius: 20px; border: 1px solid var(--border-color); margin-bottom: 3rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <span style="font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 2px; font-size: 0.8em;">Aggregate Intelligence Mastery</span>
                        <span style="font-size: 2.5em; font-weight: 800; color: #10b981;">${Math.round(totalAverage * 100)}%</span>
                    </div>
                    <div style="position: relative; height: 16px; background: var(--bg-tertiary); border-radius: 8px; overflow: hidden; border: 1px solid var(--border-color);">
                        <div style="width: ${totalAverage * 100}%; height: 100%; background: linear-gradient(90deg, #10b981, #3b82f6); border-radius: 8px; transition: width 2s cubic-bezier(0.34, 1.56, 0.64, 1);"></div>
                        <!-- Slider "Handle" Marker -->
                        <div style="position: absolute; left: calc(${totalAverage * 100}% - 8px); top: 0; width: 16px; height: 16px; background: white; border-radius: 50%; box-shadow: 0 0 10px rgba(0,0,0,0.5); z-index: 2;"></div>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 2.5rem;">
                    ${benchmarks.map(b => {
                        const score = b.last_results ? b.last_results.overall_average * 100 : 0;
                        return `
                            <div class="benchmark-card" style="background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 20px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                                    <div>
                                        <h3 style="margin: 0; font-size: 1.2em; font-weight: 700;">${b.name}</h3>
                                        <span style="font-size: 0.75em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px;">Universal Mastery</span>
                                    </div>
                                    <div style="width: 50px; height: 50px; border-radius: 50%; border: 3px solid #10b981; display: flex; align-items: center; justify-content: center; font-weight: 800; color: #10b981;">
                                        ${Math.round(score)}%
                                    </div>
                                </div>
                                <div style="height: 40px; display: flex; align-items: flex-end; gap: 3px; margin-bottom: 20px;">
                                    ${Array.from({length: 24}).map((_, i) => `
                                        <div style="flex: 1; background: ${i < (score/100 * 24) ? 'var(--accent-primary)' : 'var(--bg-tertiary)'}; height: ${30 + Math.random() * 70}%; border-radius: 1px; transition: all 1s ${i * 0.05}s ease-out;"></div>
                                    `).join('')}
                                </div>
                                <p style="font-size: 0.9em; line-height: 1.5; color: var(--text-muted); margin-bottom: 20px;">${b.description}</p>
                                <div style="display: flex; gap: 10px;">
                                    <button class="sidebar-add-btn" style="flex: 1; margin: 0; height: 36px; font-size: 11px;" onclick="window.gauntletDashboard.showModelSelector('${b.id}')">RERUN MODULE</button>
                                    <button class="sidebar-add-btn" style="flex: 0.5; margin: 0; height: 36px; font-size: 11px; background: var(--bg-tertiary);">LOGS</button>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;

        container.innerHTML = content;
        container.classList.add('active');
        document.getElementById('brain-visualizer-container')?.classList.remove('active');
    }
}

window.GauntletDashboard = GauntletDashboard;
