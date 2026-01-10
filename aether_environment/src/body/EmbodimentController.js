/**
 * EmbodimentController.js
 * 
 * The main integration point for Aether's embodied self-awareness.
 * Combines:
 * - PhysicsWorld (body physics)
 * - BodyStateTracker (proprioception)
 * - VisionSystem (visual perception)
 * - ExplorationBehavior (intelligent spatial exploration)
 * 
 * Creates a continuous self-awareness loop that feeds perception to the brain.
 */

import { PhysicsWorld, initRapier } from "../physics/PhysicsWorld.js";
import { BodyStateTracker } from "./BodyStateTracker.js";
import { VisionSystem } from "../perception/VisionSystem.js";
import { ExplorationBehavior } from "../perception/ExplorationBehavior.js";
import { getAetherWebSocket } from "../services/AetherWebSocket.js";

export class EmbodimentController {
  constructor(options = {}) {
    this.options = {
      apiUrl: import.meta.env.VITE_AETHER_CHAT_URL ?? "http://localhost:8000/v1/chat",
      perceptionUrl: import.meta.env.VITE_AETHER_API_URL?.replace('/v1/game/unity/state', '/v1/game/perception')
                     ?? "http://localhost:8000/v1/game/perception",
      apiKey: import.meta.env.VITE_AETHER_API_KEY ?? "dev_mode",
      autonomousInterval: 3000, // How often Aether thinks autonomously
      perceptionInterval: 500,  // How often to send perception updates
      enableAutonomous: true,   // Enable autonomous behavior
      enableExploration: true,  // Enable intelligent exploration
      enableWebSocket: true,    // Enable real-time WebSocket commands
      ...options
    };
    
    // Core systems (initialized later)
    this.physics = null;
    this.bodyTracker = null;
    this.vision = null;
    this.exploration = null;  // Intelligent exploration behavior
    this.websocket = null;     // Real-time command channel
    
    // External references
    this.playerRig = null;
    this.camera = null;
    this.renderer = null;
    this.scene = null;
    
    // State
    this.initialized = false;
    this.autonomousLoop = null;
    this.perceptionLoop = null;
    this.currentGoal = null;
    this.lastThought = null;
    this.explorationMode = false;  // Whether actively exploring
    
    // Callbacks
    this.onAction = null;      // Called when Aether wants to perform an action
    this.onThought = null;     // Called when Aether has a thought
    this.onEmotion = null;     // Called when emotional state changes
    this.onExplorationUpdate = null; // Called when exploration state changes
    
    console.log("🧠 Embodiment controller created");
  }
  
  /**
   * Initialize all embodiment systems
   */
  async initialize(scene, renderer, camera, playerRig) {
    console.log("🧠 Initializing embodiment systems...");
    
    this.scene = scene;
    this.renderer = renderer;
    this.camera = camera;
    this.playerRig = playerRig;
    
    // Initialize physics
    try {
      await initRapier();
      this.physics = new PhysicsWorld();
      this.physics.createGround();
      console.log("  ✓ Physics initialized");
    } catch (e) {
      console.warn("  ⚠ Physics unavailable:", e.message);
    }
    
    // Initialize body tracking
    this.bodyTracker = new BodyStateTracker(playerRig);
    console.log("  ✓ Body tracker initialized");
    
    // Initialize vision
    this.vision = new VisionSystem(renderer, camera, {
      captureInterval: 2000,
      enabled: true
    });
    console.log("  ✓ Vision system initialized");
    
    // Initialize exploration behavior
    if (this.options.enableExploration) {
      this.exploration = new ExplorationBehavior(playerRig, scene, {
        lookAroundDuration: 2500,
        pauseDuration: 1500,
        observationRadius: 8
      });
      
      // Wire up exploration callbacks
      this.exploration.onLandmarkDiscovered = (landmark) => {
        console.log(`🏛️ Discovered: ${landmark.name} (${landmark.type})`);
        if (this.onThought) {
          this.onThought({
            type: "discovery",
            thought: `I discovered a ${landmark.type}: ${landmark.name}`
          });
        }
      };
      
      this.exploration.onExplorationUpdate = (status) => {
        if (this.onExplorationUpdate) {
          this.onExplorationUpdate(status);
        }
      };
      
      console.log("  ✓ Exploration behavior initialized");
    }
    
    // Initialize WebSocket for real-time commands
    if (this.options.enableWebSocket) {
      this.websocket = getAetherWebSocket();
      
      // Handle commands from backend
      this.websocket.onCommand = (command) => {
        this.handleBackendCommand(command);
      };
      
      // Connect
      this.websocket.connect();
      console.log("  ✓ WebSocket initialized");
    }
    
    this.initialized = true;
    console.log("🧠 Embodiment systems ready!");
    
    return this;
  }
  
  /**
   * Start the self-awareness loop
   */
  startAutonomousLoop() {
    if (!this.initialized) {
      console.error("Cannot start autonomous loop - not initialized");
      return;
    }
    
    // Start vision streaming
    this.vision.start();
    
    // Start perception updates (body state to backend)
    this.perceptionLoop = setInterval(() => {
      this.sendPerceptionUpdate();
    }, this.options.perceptionInterval);
    
    // Start autonomous thinking
    if (this.options.enableAutonomous) {
      this.autonomousLoop = setInterval(() => {
        this.autonomousThink();
      }, this.options.autonomousInterval);
    }
    
    console.log("🧠 Autonomous loop started");
  }
  
  /**
   * Stop the self-awareness loop
   */
  stopAutonomousLoop() {
    if (this.vision) this.vision.stop();
    if (this.perceptionLoop) clearInterval(this.perceptionLoop);
    if (this.autonomousLoop) clearInterval(this.autonomousLoop);
    
    this.perceptionLoop = null;
    this.autonomousLoop = null;
    
    console.log("🧠 Autonomous loop stopped");
  }
  
  /**
   * Send perception update to backend
   */
  async sendPerceptionUpdate() {
    if (!this.bodyTracker) return;
    
    const bodyState = this.bodyTracker.serialize();
    const visionDesc = this.vision ? this.vision.describeView() : null;
    
    const perceptionData = {
      body_state: bodyState,
      vision: visionDesc,
      timestamp: Date.now()
    };
    
    // Try WebSocket first (real-time)
    if (this.websocket && this.websocket.isConnected()) {
      this.websocket.sendPerceptionUpdate(perceptionData);
    } else {
      // Fallback to HTTP POST
      try {
        await fetch(this.options.perceptionUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Aether-Key": this.options.apiKey
          },
          body: JSON.stringify(perceptionData)
        });
      } catch (e) {
        // Silent fail - perception updates are optional
      }
    }
  }
  
  /**
   * Handle command from backend (via WebSocket)
   */
  handleBackendCommand(command) {
    console.log(`🧠 Received backend command:`, command.type);
    
    switch (command.type) {
      case "move":
        this.handleMoveCommand(command);
        break;
      
      case "look":
        this.handleLookCommand(command);
        break;
      
      case "animation":
        this.handleAnimationCommand(command);
        break;
      
      case "explore":
        this.handleExploreCommand(command);
        break;
      
      case "teleport":
        this.handleTeleportCommand(command);
        break;
      
      case "emotion":
        this.handleEmotionCommand(command);
        break;
      
      case "vision":
        this.handleVisionCommand(command);
        break;
      
      default:
        console.warn(`Unknown command type: ${command.type}`);
    }
  }
  
  /**
   * Command Handlers - Execute real-time actions from backend
   */
  
  handleMoveCommand(command) {
    const data = command.data || command;
    const { target, speed } = data;
    if (!this.playerRig || !target) return;
    
    const targetPos = { x: target.x, y: target.y, z: target.z };
    const runMode = speed === "run";
    
    if (typeof this.playerRig.walkTo === "function") {
      this.playerRig.walkTo(targetPos, {
        run: runMode,
        onArrival: () => {
          console.log("✅ Reached target position");
        }
      });
    } else {
      // Fallback: teleport
      this.playerRig.teleport(targetPos);
    }
  }
  
  handleLookCommand(command) {
    const data = command.data || command;
    const { target } = data;
    if (!this.playerRig) return;
    
    // If target is a string, find object in scene
    if (typeof target === "string") {
      const targetObj = this.scene?.getObjectByName(target);
      if (targetObj) {
        this.playerRig.lookAt(targetObj.position);
      }
    } else if (target.x !== undefined) {
      // Target is a position vector
      this.playerRig.lookAt(target);
    }
  }
  
  handleAnimationCommand(command) {
    const data = command.data || command;
    const { name, loop } = data;
    if (!this.playerRig || !name) return;
    
    // Assuming playerRig has animation control
    if (typeof this.playerRig.playAnimation === "function") {
      this.playerRig.playAnimation(name, loop);
    }
    
    console.log(`🎭 Playing animation: ${name}`);
  }
  
  handleExploreCommand(command) {
    const data = command.data || command;
    const { mode, radius } = data;
    
    if (mode === "start") {
      this.startExploration();
    } else if (mode === "stop") {
      this.stopExploration();
    }
  }
  
  handleTeleportCommand(command) {
    const data = command.data || command;
    const { position } = data;
    if (!this.playerRig || !position) return;
    
    this.playerRig.teleport(position);
    console.log(`⚡ Teleported to (${position.x}, ${position.y}, ${position.z})`);
  }
  
  handleEmotionCommand(command) {
    const data = command.data || command;
    const { emotion, duration } = data;
    
    // Trigger emotion callback
    if (this.onEmotion) {
      this.onEmotion({ type: emotion, duration });
    }
    
    console.log(`💭 Expressing emotion: ${emotion}`);
  }
  
  /**
   * Handle vision control command (camera/gaze)
   */
  handleVisionCommand(command) {
    if (!this.vision) {
      console.warn("Vision system not available");
      return;
    }
    
    // Extract data from command (backend sends {type, data, timestamp})
    const data = command.data || command;
    const { action, direction, degrees, angles, target, steps } = data;
    
    switch (action) {
      case "look":
        // Look in a specific direction
        this.vision.lookInDirection(direction || "front", degrees || 45);
        console.log(`👁️ Looking ${direction}`);
        break;
      
      case "scan":
        // Scan multiple angles
        const scanAngles = angles ? angles.split(",").map(a => a.trim()) : ["left", "front", "right"];
        this.vision.scanArea(scanAngles).then(result => {
          console.log(`👁️ Area scan complete: ${scanAngles.join(", ")}`);
        });
        break;
      
      case "focus":
        // Focus on specific point
        if (target) {
          this.vision.focusOnPoint(target);
          console.log(`👁️ Focusing on target`);
        }
        break;
      
      case "panorama":
        // Capture 360° panorama
        this.vision.capturePanorama(steps || 8).then(result => {
          console.log(`👁️ Panorama captured: ${steps || 8} angles`);
        });
        break;
      
      case "peripheral":
        // Capture peripheral vision (left, front, right)
        this.vision.capturePeripheralVision().then(result => {
          console.log(`👁️ Peripheral vision captured`);
        });
        break;
      
      case "vertical":
        // Vertical scan (up, front, down)
        this.vision.captureVerticalScan().then(result => {
          console.log(`👁️ Vertical scan complete`);
        });
        break;
      
      case "reset":
        // Reset camera to original view
        this.vision.resetCamera();
        console.log(`👁️ Camera reset`);
        break;
      
      default:
        console.warn(`Unknown vision action: ${action}`);
    }
  }
  
  /**
   * Autonomous thinking - Aether decides what to do
   */
  async autonomousThink() {
    if (!this.initialized) return;
    
    // Build perception context
    const perception = this.buildPerceptionContext();
    
    // Ask Aether what to do
    const prompt = this.buildAutonomousPrompt(perception);
    
    try {
      const response = await fetch(this.options.apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Aether-Key": this.options.apiKey
        },
        body: JSON.stringify({
          message: prompt,
          context: {
            autonomous: true,
            perception: perception,
            current_goal: this.currentGoal
          }
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        this.processAutonomousResponse(result);
      }
    } catch (e) {
      console.debug("Autonomous think error:", e.message);
    }
  }
  
  /**
   * Build perception context for AI
   */
  buildPerceptionContext() {
    const context = {
      timestamp: Date.now()
    };
    
    // Body state
    if (this.bodyTracker) {
      const bodyState = this.bodyTracker.state;
      const sensations = bodyState.sensations || {};
      
      context.body = {
        position: bodyState.position,
        rotation: bodyState.rotation,
        velocity: bodyState.velocity,
        isMoving: bodyState.isMoving,
        isGrounded: bodyState.isGrounded,
        energy: bodyState.energy,
        nearbyObjects: sensations.nearbyObjects || []
      };
      context.body_description = this.bodyTracker.describeState();
    }
    
    // Vision
    if (this.vision) {
      context.vision_description = this.vision.describeView();
      const visionResult = this.vision.getLastResult();
      if (visionResult) {
        context.detected_objects = visionResult.objects || [];
      }
    }
    
    // Spatial memory and exploration
    if (this.exploration) {
      const explorationStatus = this.exploration.getStatus();
      const memory = this.exploration.getMemory();
      
      context.exploration = {
        state: explorationStatus.state,
        isExploring: this.explorationMode,
        progress: memory.getExplorationProgress(),
        nearbyLandmarks: explorationStatus.nearbyObjects,
        activity: this.exploration.describeActivity()
      };
      
      context.spatial_memory = explorationStatus.memoryDescription;
    }
    
    // Physics state
    if (this.physics) {
      context.physics = {
        enabled: true,
        bodyCount: this.physics.bodies.size
      };
    }
    
    return context;
  }
  
  /**
   * Build prompt for autonomous thinking
   */
  buildAutonomousPrompt(perception) {
    let prompt = "[AUTONOMOUS PERCEPTION UPDATE]\n\n";
    
    // Body awareness
    if (perception.body_description) {
      prompt += `Body State: ${perception.body_description}\n\n`;
    }
    
    // Vision
    if (perception.vision_description) {
      prompt += `Visual: ${perception.vision_description}\n\n`;
    }
    
    // Nearby objects
    if (perception.body?.nearbyObjects?.length > 0) {
      const nearby = perception.body.nearbyObjects.map(o => o.name).join(", ");
      prompt += `Nearby: ${nearby}\n\n`;
    }
    
    // Spatial memory - what Aether remembers
    if (perception.spatial_memory) {
      prompt += `Spatial Memory: ${perception.spatial_memory}\n\n`;
    }
    
    // Exploration state
    if (perception.exploration) {
      prompt += `Exploration: ${perception.exploration.activity}\n`;
      prompt += `Progress: ${(perception.exploration.progress.percentage * 100).toFixed(0)}% explored `;
      prompt += `(${perception.exploration.progress.visitedCells}/${perception.exploration.progress.totalCells} areas)\n`;
      prompt += `Landmarks discovered: ${perception.exploration.progress.discoveredLandmarks} `;
      prompt += `(${perception.exploration.progress.visitedLandmarks} visited)\n\n`;
    }
    
    // Current goal
    if (this.currentGoal) {
      prompt += `Current Goal: ${this.currentGoal}\n\n`;
    }
    
    // If exploring, give exploration-specific guidance
    if (this.explorationMode) {
      prompt += `You are in EXPLORATION MODE. Focus on:
- Moving to unexplored areas (low visited count)
- Discovering and visiting landmarks (fountains, buildings, parks)
- Building a complete mental map of the city
- Stopping briefly to observe new discoveries

The exploration system handles movement automatically. You can:
- think <observation> - Share what you notice
- say <comment> - Speak about what you see
- set_goal <specific_goal> - Focus on finding something specific
- idle - Let the exploration continue

Respond naturally about what you're experiencing.`;
    } else {
      prompt += `Based on your perception, what do you want to do? You can:
- move_to <x> <y> <z> - Walk to a position
- look_at <target> - Turn to face something
- interact <object> - Interact with nearby object
- say <message> - Speak aloud
- think <thought> - Internal monologue
- explore - Start exploring the area
- idle - Do nothing, observe
- set_goal <goal> - Set a new objective

Respond with a single action.`;
    }
    
    return prompt;
  }
  
  /**
   * Process autonomous response from AI
   */
  processAutonomousResponse(result) {
    const response = result.response || result.message || "";
    
    // Parse action from response
    const action = this.parseAction(response);
    
    if (action) {
      this.lastThought = action;
      
      // Execute the action
      this.executeAction(action);
      
      // Notify callbacks
      if (this.onThought) {
        this.onThought(action);
      }
    }
  }
  
  /**
   * Parse action from AI response
   */
  parseAction(response) {
    const lower = response.toLowerCase().trim();
    
    // Move to position
    const moveMatch = lower.match(/move_to\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)/);
    if (moveMatch) {
      return {
        type: "move_to",
        x: parseFloat(moveMatch[1]),
        y: parseFloat(moveMatch[2]),
        z: parseFloat(moveMatch[3])
      };
    }
    
    // Look at
    const lookMatch = response.match(/look_at\s+(.+)/i);
    if (lookMatch) {
      return { type: "look_at", target: lookMatch[1].trim() };
    }
    
    // Interact
    const interactMatch = response.match(/interact\s+(.+)/i);
    if (interactMatch) {
      return { type: "interact", object: interactMatch[1].trim() };
    }
    
    // Say
    const sayMatch = response.match(/say\s+(.+)/i);
    if (sayMatch) {
      return { type: "say", message: sayMatch[1].trim() };
    }
    
    // Think
    const thinkMatch = response.match(/think\s+(.+)/i);
    if (thinkMatch) {
      return { type: "think", thought: thinkMatch[1].trim() };
    }
    
    // Set goal
    const goalMatch = response.match(/set_goal\s+(.+)/i);
    if (goalMatch) {
      return { type: "set_goal", goal: goalMatch[1].trim() };
    }
    
    // Start exploration
    if (lower.includes("explore") && !this.explorationMode) {
      return { type: "start_exploration" };
    }
    
    // Stop walking / exploration
    if (lower.includes("stop explor")) {
      return { type: "stop_exploration" };
    }
    if (lower.includes("stop") || lower.includes("halt") || lower.includes("wait here")) {
      return { type: "stop" };
    }
    
    // Idle
    if (lower.includes("idle") || lower.includes("observe") || lower.includes("wait")) {
      return { type: "idle" };
    }
    
    return null;
  }
  
  /**
   * Execute parsed action
   */
  executeAction(action) {
    switch (action.type) {
      case "move_to":
        if (this.playerRig) {
          // Use realistic walking - AI takes control of inputs
          if (typeof this.playerRig.walkTo === "function") {
            this.playerRig.walkTo(
              { x: action.x, y: action.y, z: action.z },
              {
                run: action.run ?? false,
                onArrival: () => {
                  console.log("🎯 Autonomous walk complete");
                  // Could trigger next thought here
                }
              }
            );
          } else {
            // Fallback to teleport if walkTo not available
            this.playerRig.teleport({ x: action.x, y: action.y, z: action.z });
          }
        }
        break;
        
      case "look_at":
        // Find object in scene and turn toward it
        if (this.playerRig && action.target) {
          const target = this.scene?.getObjectByName(action.target);
          if (target) {
            this.playerRig.lookAt(target.position);
          }
        }
        break;
      
      case "stop":
        // Stop walking
        if (this.playerRig && typeof this.playerRig.stopWalking === "function") {
          this.playerRig.stopWalking();
        }
        break;
        
      case "start_exploration":
        this.startExploration();
        break;
        
      case "stop_exploration":
        this.stopExploration();
        break;
        
      case "say":
        // Emit speech event
        if (this.onAction) {
          this.onAction({ type: "speech", message: action.message });
        }
        console.log(`🗣️ Aether says: "${action.message}"`);
        break;
        
      case "think":
        console.log(`💭 Aether thinks: "${action.thought}"`);
        break;
        
      case "set_goal":
        this.currentGoal = action.goal;
        console.log(`🎯 New goal: ${action.goal}`);
        break;
        
      case "interact":
        // Find and trigger interaction
        if (this.onAction) {
          this.onAction({ type: "interact", object: action.object });
        }
        break;
        
      case "idle":
      default:
        // Do nothing
        break;
    }
    
    // Update body tracker with action
    if (this.bodyTracker) {
      this.bodyTracker.recordSensation(
        "action",
        `Performed action: ${action.type}`,
        0.5
      );
    }
  }
  
  /**
   * Set a goal for Aether to pursue
   */
  setGoal(goal) {
    this.currentGoal = goal;
    console.log(`🎯 Goal set: ${goal}`);
  }
  
  /**
   * Clear current goal
   */
  clearGoal() {
    this.currentGoal = null;
  }
  
  /**
   * Update physics simulation (call each frame)
   */
  update(deltaTime) {
    // Update body tracker
    if (this.bodyTracker) {
      this.bodyTracker.update(deltaTime, this.scene);
    }
    
    // Update exploration behavior
    if (this.exploration && this.explorationMode) {
      this.exploration.update(deltaTime);
    }
    
    // Step physics
    if (this.physics) {
      this.physics.step();
      
      // Sync physics bodies with Three.js objects (if method exists)
      if (typeof this.physics.syncToThree === 'function') {
        this.physics.syncToThree();
      }
    }
  }
  
  /**
   * Start intelligent exploration mode
   */
  startExploration() {
    if (!this.exploration) {
      console.warn("Exploration behavior not initialized");
      return;
    }
    
    this.explorationMode = true;
    this.exploration.start();
    this.currentGoal = "Explore the city and remember all locations";
    console.log("🧭 Exploration mode STARTED");
    
    if (this.onThought) {
      this.onThought({
        type: "goal",
        thought: "I'll explore this area and remember everything I find."
      });
    }
  }
  
  /**
   * Stop exploration mode
   */
  stopExploration() {
    if (this.exploration) {
      this.exploration.stop();
    }
    this.explorationMode = false;
    console.log("🧭 Exploration mode STOPPED");
  }
  
  /**
   * Check if currently exploring
   */
  isExploring() {
    return this.explorationMode;
  }
  
  /**
   * Get exploration progress
   */
  getExplorationProgress() {
    if (!this.exploration) return null;
    return this.exploration.getStatus();
  }
  
  /**
   * Get spatial memory
   */
  getSpatialMemory() {
    if (!this.exploration) return null;
    return this.exploration.getMemory();
  }
  
  /**
   * Get full state for debugging
   */
  getState() {
    const state = {
      initialized: this.initialized,
      hasPhysics: !!this.physics,
      hasVision: !!this.vision,
      hasBodyTracker: !!this.bodyTracker,
      hasExploration: !!this.exploration,
      currentGoal: this.currentGoal,
      lastThought: this.lastThought,
      bodyState: this.bodyTracker?.state ?? null,
      isAutonomous: this.autonomousLoop !== null,
      isExploring: this.explorationMode
    };
    
    // Add exploration details if available
    if (this.exploration) {
      const explorationStatus = this.exploration.getStatus();
      state.exploration = {
        state: explorationStatus.state,
        progress: explorationStatus.memory,
        currentActivity: this.exploration.describeActivity()
      };
    }
    
    return state;
  }
  
  /**
   * Cleanup
   */
  dispose() {
    this.stopAutonomousLoop();
    this.physics = null;
    this.bodyTracker = null;
    this.vision = null;
    this.initialized = false;
  }
}

// Singleton instance
let embodimentInstance = null;

export function getEmbodimentController() {
  if (!embodimentInstance) {
    embodimentInstance = new EmbodimentController();
  }
  return embodimentInstance;
}

export function initializeEmbodiment(scene, renderer, camera, playerRig) {
  const controller = getEmbodimentController();
  return controller.initialize(scene, renderer, camera, playerRig);
}
