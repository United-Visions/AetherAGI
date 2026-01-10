/**
 * AetherMind Bridge for PlayCanvas
 * 
 * Browser-based game engine integration - No installation required!
 * Works in PlayCanvas Editor: https://playcanvas.com
 * 
 * Setup Instructions:
 * 1. Go to playcanvas.com and create a free account
 * 2. Create a new project
 * 3. Create a new Script Asset, paste this code
 * 4. Attach to any Entity in your scene (e.g., Player or GameManager)
 * 5. Configure the API URL in the inspector
 * 6. Press Launch to test!
 */

var AetherBridge = pc.createScript('aetherBridge');

// Script Attributes (appear in PlayCanvas Inspector)
AetherBridge.attributes.add('apiUrl', {
    type: 'string',
    default: 'http://localhost:8000/v1/game/unity/state',
    title: 'AetherMind API URL',
    description: 'Backend endpoint for game state sync'
});

AetherBridge.attributes.add('syncInterval', {
    type: 'number',
    default: 500,
    title: 'Sync Interval (ms)',
    description: 'How often to sync with AetherMind (milliseconds)'
});

AetherBridge.attributes.add('playerEntity', {
    type: 'entity',
    title: 'Player Entity',
    description: 'Reference to the player entity'
});

AetherBridge.attributes.add('debugMode', {
    type: 'boolean',
    default: true,
    title: 'Debug Mode',
    description: 'Log messages to console'
});

// Initialize
AetherBridge.prototype.initialize = function() {
    this.eventQueue = [];
    this.pendingCommands = [];
    this.isConnected = false;
    
    // Start sync loop
    this.syncTimer = setInterval(() => {
        this.syncWithAether();
    }, this.syncInterval);
    
    this.log('🧠 AetherMind Bridge initialized');
    this.log(`Connecting to: ${this.apiUrl}`);
};

// Main sync function - sends state, receives commands
AetherBridge.prototype.syncWithAether = function() {
    // Gather game state
    const gameState = this.collectGameState();
    
    // Send to AetherMind backend
    fetch(this.apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Aether-Key': 'dev_mode'  // For local dev
        },
        body: JSON.stringify(gameState)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (!this.isConnected) {
            this.isConnected = true;
            this.log('✅ Connected to AetherMind backend', 'success');
        }
        
        // Execute commands from AetherMind
        if (data.commands && data.commands.length > 0) {
            this.log(`🧠 Received ${data.commands.length} command(s)`, 'info');
            data.commands.forEach(cmd => this.executeCommand(cmd));
        }
        
        // Clear processed events
        this.eventQueue = [];
    })
    .catch(error => {
        if (this.isConnected) {
            this.log(`⚠️ Connection lost: ${error.message}`, 'warning');
            this.isConnected = false;
        }
    });
};

// Collect current game state
AetherBridge.prototype.collectGameState = function() {
    const state = {
        timestamp: Date.now(),
        currentScene: this.app.scene.name || 'Untitled',
        events: [...this.eventQueue],
        entities: {}
    };
    
    // Player state
    if (this.playerEntity) {
        const pos = this.playerEntity.getPosition();
        const rot = this.playerEntity.getRotation().getEulerAngles();
        
        state.entities.player = {
            position: { x: pos.x, y: pos.y, z: pos.z },
            rotation: { x: rot.x, y: rot.y, z: rot.z },
            active: this.playerEntity.enabled
        };
    }
    
    // Count entities by tag
    state.entityCounts = {
        enemies: this.app.root.findByTag('enemy').length,
        collectibles: this.app.root.findByTag('collectible').length,
        npcs: this.app.root.findByTag('npc').length
    };
    
    // Performance metrics
    state.performance = {
        fps: Math.round(this.app.stats.fps),
        drawCalls: this.app.stats.drawCalls,
        triangles: this.app.stats.triangles
    };
    
    return state;
};

// Execute command from AetherMind
AetherBridge.prototype.executeCommand = function(cmd) {
    this.log(`⚡ Executing: ${cmd.action} on ${cmd.target || 'scene'}`, 'info');
    
    const params = cmd.params || {};
    
    switch (cmd.action) {
        case 'move':
            this.handleMove(cmd.target, params);
            break;
            
        case 'spawn':
            this.handleSpawn(params);
            break;
            
        case 'destroy':
            this.handleDestroy(cmd.target);
            break;
            
        case 'set_time':
            this.handleSetTime(params);
            break;
            
        case 'chat':
            this.handleChat(params);
            break;
            
        case 'camera_switch':
            this.handleCameraSwitch(params);
            break;
            
        case 'set_property':
            this.handleSetProperty(cmd.target, params);
            break;
            
        default:
            this.log(`⚠️ Unknown action: ${cmd.action}`, 'warning');
            break;
    }
};

// Command Handlers

AetherBridge.prototype.handleMove = function(target, params) {
    let entity = null;
    
    if (target === 'player' && this.playerEntity) {
        entity = this.playerEntity;
    } else {
        entity = this.app.root.findByName(target);
    }
    
    if (entity) {
        const x = params.x !== undefined ? params.x : entity.getPosition().x;
        const y = params.y !== undefined ? params.y : entity.getPosition().y;
        const z = params.z !== undefined ? params.z : entity.getPosition().z;
        
        entity.setPosition(x, y, z);
        this.log(`Moved ${target} to (${x}, ${y}, ${z})`);
    } else {
        this.log(`Entity not found: ${target}`, 'warning');
    }
};

AetherBridge.prototype.handleSpawn = function(params) {
    const type = params.type || 'default';
    const pos = params.position || { x: 0, y: 0, z: 0 };
    
    // Create a simple entity (customize based on your game)
    const entity = new pc.Entity(type);
    entity.addComponent('model', { type: 'box' });
    entity.setPosition(pos.x, pos.y, pos.z);
    entity.tags.add(type);
    
    this.app.root.addChild(entity);
    this.log(`Spawned ${type} at (${pos.x}, ${pos.y}, ${pos.z})`);
};

AetherBridge.prototype.handleDestroy = function(target) {
    const entity = this.app.root.findByName(target);
    if (entity) {
        entity.destroy();
        this.log(`Destroyed ${target}`);
    }
};

AetherBridge.prototype.handleSetTime = function(params) {
    const hour = params.hour || 12;
    
    // Example: adjust directional light intensity based on time
    const lights = this.app.root.findByTag('sun');
    lights.forEach(light => {
        if (light.light) {
            // Day (6-18) = bright, Night = dim
            const intensity = (hour >= 6 && hour <= 18) ? 1.0 : 0.1;
            light.light.intensity = intensity;
        }
    });
    
    this.log(`Set time to ${hour}:00`);
};

AetherBridge.prototype.handleChat = function(params) {
    const message = params.message || params.text || 'Hello!';
    
    // Display in console or your game UI
    this.log(`💬 AetherMind says: ${message}`, 'success');
    
    // If you have a UI system, show the message there
    this.app.fire('aether:chat', message);
};

AetherBridge.prototype.handleCameraSwitch = function(params) {
    const cameraName = params.camera || params.name;
    const camera = this.app.root.findByName(cameraName);
    
    if (camera && camera.camera) {
        // Disable all cameras
        this.app.root.findByTag('camera').forEach(cam => {
            if (cam.camera) cam.camera.enabled = false;
        });
        
        // Enable target camera
        camera.camera.enabled = true;
        this.log(`Switched to camera: ${cameraName}`);
    }
};

AetherBridge.prototype.handleSetProperty = function(target, params) {
    const entity = this.app.root.findByName(target);
    if (!entity) return;
    
    // Set arbitrary properties
    if (params.enabled !== undefined) {
        entity.enabled = params.enabled;
    }
    if (params.scale !== undefined) {
        entity.setLocalScale(params.scale, params.scale, params.scale);
    }
    if (params.color !== undefined && entity.model) {
        const material = entity.model.meshInstances[0].material;
        material.diffuse.set(params.color.r, params.color.g, params.color.b);
        material.update();
    }
    
    this.log(`Updated properties for ${target}`);
};

// Public API - Log game events that AetherMind should know about
AetherBridge.prototype.logEvent = function(eventDescription) {
    this.eventQueue.push({
        timestamp: Date.now(),
        description: eventDescription
    });
    this.log(`📝 Event logged: ${eventDescription}`);
};

// Logging utility
AetherBridge.prototype.log = function(message, level = 'info') {
    if (!this.debugMode) return;
    
    const prefix = {
        info: '🔵',
        success: '✅',
        warning: '⚠️',
        error: '❌'
    }[level] || '🔵';
    
    console.log(`${prefix} [AetherBridge] ${message}`);
};

// Cleanup
AetherBridge.prototype.destroy = function() {
    if (this.syncTimer) {
        clearInterval(this.syncTimer);
    }
    this.log('AetherBridge destroyed');
};

// Update loop (optional - for additional logic)
AetherBridge.prototype.update = function(dt) {
    // Add any per-frame logic here if needed
};

// Example: Trigger event on collision
AetherBridge.prototype.onCollisionStart = function(result) {
    this.logEvent(`Collision: ${this.entity.name} hit ${result.other.name}`);
};

/**
 * USAGE EXAMPLES (from other scripts):
 * 
 * // Get the bridge script
 * const bridge = this.app.root.findByName('GameManager').script.aetherBridge;
 * 
 * // Log custom events
 * bridge.logEvent('Player collected coin');
 * bridge.logEvent('Boss spawned at wave 5');
 * bridge.logEvent('Player health low: 20%');
 * 
 * // Listen for AetherMind chat messages
 * this.app.on('aether:chat', function(message) {
 *     console.log('AetherMind:', message);
 *     // Display in your game UI
 * });
 */
