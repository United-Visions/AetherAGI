// BackgroundTaskManager.js - Jules-style persistent task management
// Tasks persist across sessions and never stop until complete

export class BackgroundTaskManager {
    constructor() {
        this.tasks = new Map();
        this.listeners = [];
        this.pollInterval = null;
        
        // Load persisted tasks
        this.load();
        
        // Start polling for updates
        this.startPolling();
    }

    load() {
        try {
            const stored = localStorage.getItem('aethermind_tasks');
            if (stored) {
                const tasks = JSON.parse(stored);
                tasks.forEach(task => this.tasks.set(task.id, task));
            }
        } catch (err) {
            console.error('Failed to load tasks:', err);
        }
        
        this.updateIndicator();
    }

    save() {
        try {
            const tasks = Array.from(this.tasks.values());
            localStorage.setItem('aethermind_tasks', JSON.stringify(tasks));
        } catch (err) {
            console.error('Failed to save tasks:', err);
        }
    }

    addTask(task) {
        const fullTask = {
            id: task.id || `task_${Date.now()}`,
            type: task.type || 'generic',
            name: task.name || 'Background Task',
            status: task.status || 'pending',
            progress: task.progress || 0,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            data: task.data || {},
            retries: 0,
            max_retries: task.max_retries || 3
        };
        
        this.tasks.set(fullTask.id, fullTask);
        this.save();
        this.updateIndicator();
        this.notifyListeners('added', fullTask);
        
        console.log(`📋 [TASKS] Added task: ${fullTask.name} (${fullTask.id})`);
        
        return fullTask.id;
    }

    updateTask(id, updates) {
        const task = this.tasks.get(id);
        if (!task) return;
        
        Object.assign(task, updates, { updated_at: new Date().toISOString() });
        this.save();
        this.updateIndicator();
        this.notifyListeners('updated', task);
        
        // If completed or errored, notify user
        if (updates.status === 'completed') {
            this.onTaskComplete(task);
        } else if (updates.status === 'error') {
            this.onTaskError(task);
        }
    }

    removeTask(id) {
        const task = this.tasks.get(id);
        if (task) {
            this.tasks.delete(id);
            this.save();
            this.updateIndicator();
            this.notifyListeners('removed', task);
        }
    }

    getTask(id) {
        return this.tasks.get(id);
    }

    getTasks(filter = null) {
        let tasks = Array.from(this.tasks.values());
        
        if (filter) {
            if (filter.status) {
                tasks = tasks.filter(t => t.status === filter.status);
            }
            if (filter.type) {
                tasks = tasks.filter(t => t.type === filter.type);
            }
        }
        
        // Sort by updated_at descending
        return tasks.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
    }

    getRunningTasks() {
        return this.getTasks({ status: 'running' });
    }

    updateIndicator() {
        const running = this.getRunningTasks().length;
        const indicator = document.getElementById('tasks-indicator');
        const count = document.getElementById('tasks-count');
        
        if (indicator && count) {
            if (running > 0) {
                indicator.classList.remove('hidden');
                count.textContent = running;
            } else {
                indicator.classList.add('hidden');
            }
        }
    }

    startPolling() {
        // Poll backend for task updates every 5 seconds
        this.pollInterval = setInterval(() => this.pollTasks(), 5000);
        
        // Also poll immediately
        this.pollTasks();
    }

    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    async pollTasks() {
        const runningTasks = this.getRunningTasks();
        if (runningTasks.length === 0) return;
        
        try {
            const apiKey = localStorage.getItem('aethermind_api_key');
            if (!apiKey) return;
            
            const response = await fetch('/v1/tasks/status', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': apiKey
                },
                body: JSON.stringify({
                    task_ids: runningTasks.map(t => t.id)
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                
                // Update local tasks with backend status
                data.tasks?.forEach(update => {
                    this.updateTask(update.id, {
                        status: update.status,
                        progress: update.progress,
                        result: update.result,
                        error: update.error
                    });
                });
            }
        } catch (err) {
            console.error('Failed to poll tasks:', err);
        }
    }

    async retryTask(id) {
        const task = this.tasks.get(id);
        if (!task) return;
        
        if (task.retries >= task.max_retries) {
            console.warn(`Task ${id} exceeded max retries`);
            return;
        }
        
        task.retries++;
        task.status = 'pending';
        task.error = null;
        this.save();
        
        // Re-submit to backend
        try {
            await fetch('/v1/tasks/retry', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Aether-Key': localStorage.getItem('aethermind_api_key')
                },
                body: JSON.stringify({ task_id: id })
            });
        } catch (err) {
            console.error('Failed to retry task:', err);
        }
    }

    onTaskComplete(task) {
        console.log(`✅ [TASKS] Task completed: ${task.name}`);
        
        // Add notification
        window.aether?.notifications?.add({
            type: 'task_complete',
            title: 'Task Completed',
            message: `${task.name} has finished successfully`,
            data: { taskId: task.id }
        });
        
        // Show toast
        window.aether?.toast(`${task.name} completed!`, 'success');
    }

    onTaskError(task) {
        console.log(`❌ [TASKS] Task failed: ${task.name}`);
        
        // Add notification
        window.aether?.notifications?.add({
            type: 'task_error',
            title: 'Task Failed',
            message: `${task.name} encountered an error: ${task.error || 'Unknown error'}`,
            data: { taskId: task.id }
        });
        
        // Auto-retry if possible
        if (task.retries < task.max_retries) {
            setTimeout(() => this.retryTask(task.id), 5000);
        }
    }

    notifyListeners(event, task) {
        this.listeners.forEach(fn => fn(event, task));
    }

    onTaskChange(callback) {
        this.listeners.push(callback);
    }

    // Create common task types
    createResearchTask(query, options = {}) {
        return this.addTask({
            type: 'research',
            name: `Researching: ${query.substring(0, 50)}...`,
            data: { query, ...options }
        });
    }

    createBuildTask(spec, options = {}) {
        return this.addTask({
            type: 'build',
            name: `Building: ${spec.name || 'project'}`,
            data: { spec, ...options }
        });
    }

    createToolForgeTask(spec) {
        return this.addTask({
            type: 'tool_forge',
            name: `Forging tool: ${spec.name}`,
            data: { spec }
        });
    }

    createGoalTask(goal) {
        return this.addTask({
            type: 'goal',
            name: goal.title || 'Autonomous Goal',
            data: { goal },
            max_retries: 5 // Goals get more retries
        });
    }
}
