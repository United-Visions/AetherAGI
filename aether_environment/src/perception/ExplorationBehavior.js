/**
 * ExplorationBehavior.js
 * 
 * Intelligent exploration behavior for Aether.
 * Makes Aether explore the city in a human-like manner:
 * - Look around before moving
 * - Choose interesting directions
 * - Stop and observe landmarks
 * - Remember what's been seen
 * - Avoid revisiting the same spots too often
 */

import * as THREE from "three";
import { SpatialMemory } from "./SpatialMemory.js";

// Exploration states
const ExploreState = {
  IDLE: "idle",
  LOOKING_AROUND: "looking_around",
  CHOOSING_DIRECTION: "choosing_direction",
  WALKING: "walking",
  OBSERVING_LANDMARK: "observing_landmark",
  PAUSING: "pausing"
};

export class ExplorationBehavior {
  constructor(playerRig, scene, options = {}) {
    this.player = playerRig;
    this.scene = scene;
    
    this.options = {
      lookAroundDuration: 2000,      // How long to look around (ms)
      pauseDuration: 1500,           // How long to pause at landmarks
      minWalkDistance: 3,            // Minimum distance per walk
      maxWalkDistance: 10,           // Maximum distance per walk
      observationRadius: 5,          // How far to detect objects
      scanInterval: 500,             // How often to scan surroundings
      ...options
    };
    
    // Spatial memory
    this.memory = new SpatialMemory({
      cellSize: 2.0,
      explorationRadius: 2.0
    });
    
    // Current state
    this.state = ExploreState.IDLE;
    this.stateStartTime = 0;
    this.currentTarget = null;
    this.lookDirection = 0;
    this.lastScanTime = 0;
    
    // Detected objects in scene
    this.nearbyObjects = [];
    this.knownLandmarks = [];
    
    // Callbacks
    this.onStateChange = null;
    this.onLandmarkDiscovered = null;
    this.onExplorationUpdate = null;
    
    console.log("🧭 Exploration behavior initialized");
  }
  
  /**
   * Start exploring
   */
  start() {
    this.setState(ExploreState.LOOKING_AROUND);
    console.log("🧭 Exploration started!");
  }
  
  /**
   * Stop exploring
   */
  stop() {
    this.setState(ExploreState.IDLE);
    if (this.player.stopWalking) {
      this.player.stopWalking();
    }
  }
  
  /**
   * Set current state
   */
  setState(newState) {
    if (this.state === newState) return;
    
    const oldState = this.state;
    this.state = newState;
    this.stateStartTime = Date.now();
    
    console.log(`🧭 State: ${oldState} → ${newState}`);
    
    if (this.onStateChange) {
      this.onStateChange(newState, oldState);
    }
  }
  
  /**
   * Update exploration (call every frame)
   */
  update(dt) {
    if (this.state === ExploreState.IDLE) return;
    
    if (!this.player.model) return;
    
    const currentPos = this.player.model.position;
    const now = Date.now();
    const stateTime = now - this.stateStartTime;
    
    // Always update spatial memory
    this.memory.visit({
      x: currentPos.x,
      y: currentPos.y,
      z: currentPos.z
    });
    
    // Periodic scanning for nearby objects
    if (now - this.lastScanTime > this.options.scanInterval) {
      this.scanSurroundings();
      this.lastScanTime = now;
    }
    
    // State machine
    switch (this.state) {
      case ExploreState.LOOKING_AROUND:
        this._updateLookingAround(dt, stateTime);
        break;
        
      case ExploreState.CHOOSING_DIRECTION:
        this._updateChoosingDirection(dt, stateTime);
        break;
        
      case ExploreState.WALKING:
        this._updateWalking(dt, stateTime);
        break;
        
      case ExploreState.OBSERVING_LANDMARK:
        this._updateObservingLandmark(dt, stateTime);
        break;
        
      case ExploreState.PAUSING:
        this._updatePausing(dt, stateTime);
        break;
    }
    
    // Notify of exploration updates
    if (this.onExplorationUpdate) {
      this.onExplorationUpdate(this.getStatus());
    }
  }
  
  /**
   * Look around state - rotate to observe surroundings
   */
  _updateLookingAround(dt, stateTime) {
    // Slowly rotate to look around
    const rotationSpeed = (Math.PI * 2) / (this.options.lookAroundDuration / 1000);
    this.lookDirection += rotationSpeed * dt;
    
    // Apply rotation to player (look direction, not movement)
    if (this.player.model) {
      // Gradually rotate
      const targetRot = this.lookDirection % (Math.PI * 2);
      let diff = targetRot - this.player.model.rotation.y;
      while (diff > Math.PI) diff -= Math.PI * 2;
      while (diff < -Math.PI) diff += Math.PI * 2;
      this.player.model.rotation.y += diff * 0.1;
    }
    
    // After looking around, choose direction
    if (stateTime > this.options.lookAroundDuration) {
      this.setState(ExploreState.CHOOSING_DIRECTION);
    }
  }
  
  /**
   * Choose where to go next
   */
  _updateChoosingDirection(dt, stateTime) {
    const currentPos = this.player.model.position;
    
    // Priority 1: Unvisited landmarks nearby
    const unvisitedLandmark = this.memory.getNearestUnvisitedLandmark(currentPos);
    if (unvisitedLandmark && unvisitedLandmark.distance < 15) {
      this.currentTarget = {
        type: "landmark",
        position: unvisitedLandmark.landmark.position,
        landmark: unvisitedLandmark.landmark
      };
      console.log(`🎯 Heading to: ${unvisitedLandmark.landmark.name}`);
      this._startWalking();
      return;
    }
    
    // Priority 2: Unexplored areas
    const unexplored = this.memory.getUnexploredTargets(currentPos, 3);
    if (unexplored.length > 0) {
      // Pick one with some randomness
      const target = unexplored[Math.floor(Math.random() * Math.min(3, unexplored.length))];
      this.currentTarget = {
        type: "exploration",
        position: { x: target.x, y: 0, z: target.z }
      };
      console.log(`🧭 Exploring toward (${target.x.toFixed(1)}, ${target.z.toFixed(1)})`);
      this._startWalking();
      return;
    }
    
    // Priority 3: Random direction (all explored!)
    const bestDir = this.memory.getBestExplorationDirection(currentPos);
    if (bestDir) {
      const distance = this.options.minWalkDistance + 
                      Math.random() * (this.options.maxWalkDistance - this.options.minWalkDistance);
      this.currentTarget = {
        type: "wander",
        position: {
          x: currentPos.x + bestDir.dx * distance,
          y: 0,
          z: currentPos.z + bestDir.dz * distance
        }
      };
      console.log(`🚶 Wandering ${bestDir.name}`);
      this._startWalking();
      return;
    }
    
    // Fallback: pause and look around again
    this.setState(ExploreState.PAUSING);
  }
  
  /**
   * Start walking to target
   */
  _startWalking() {
    if (!this.currentTarget || !this.player.walkTo) {
      this.setState(ExploreState.LOOKING_AROUND);
      return;
    }
    
    this.player.walkTo(this.currentTarget.position, {
      run: this.currentTarget.type === "landmark",
      onArrival: () => this._onArrival()
    });
    
    this.setState(ExploreState.WALKING);
  }
  
  /**
   * Walking state - monitor progress
   */
  _updateWalking(dt, stateTime) {
    // Check if we're still walking
    if (!this.player.isWalking || !this.player.isWalking()) {
      // Arrived or stopped
      this._onArrival();
      return;
    }
    
    // Check for interesting objects along the way
    const nearbyInteresting = this.nearbyObjects.filter(obj => {
      // Is this a landmark we haven't visited?
      const landmark = this.memory.landmarks.get(obj.id);
      return landmark && !landmark.visited && obj.distance < 2;
    });
    
    if (nearbyInteresting.length > 0) {
      // Stop to observe!
      this.player.stopWalking();
      this.currentTarget = {
        type: "landmark",
        landmark: this.memory.landmarks.get(nearbyInteresting[0].id)
      };
      this.setState(ExploreState.OBSERVING_LANDMARK);
    }
    
    // Timeout protection
    if (stateTime > 30000) {
      console.log("⚠️ Walk timeout, resetting");
      this.player.stopWalking();
      this.setState(ExploreState.LOOKING_AROUND);
    }
  }
  
  /**
   * Called when arriving at destination
   */
  _onArrival() {
    if (this.currentTarget?.type === "landmark" && this.currentTarget.landmark) {
      // Mark landmark as visited
      this.memory.visitLandmark(this.currentTarget.landmark.id);
      console.log(`✅ Visited: ${this.currentTarget.landmark.name}`);
      
      // Observe the landmark
      this.setState(ExploreState.OBSERVING_LANDMARK);
    } else {
      // Just paused, then look around
      this.setState(ExploreState.PAUSING);
    }
    
    this.currentTarget = null;
  }
  
  /**
   * Observing a landmark
   */
  _updateObservingLandmark(dt, stateTime) {
    // Look at the landmark (slowly rotate)
    if (this.currentTarget?.landmark) {
      const landmark = this.currentTarget.landmark;
      const pos = this.player.model.position;
      const targetAngle = Math.atan2(
        landmark.position.x - pos.x,
        landmark.position.z - pos.z
      );
      
      let diff = targetAngle - this.player.model.rotation.y;
      while (diff > Math.PI) diff -= Math.PI * 2;
      while (diff < -Math.PI) diff += Math.PI * 2;
      this.player.model.rotation.y += diff * 0.05;
    }
    
    // After observing, look around for more
    if (stateTime > this.options.pauseDuration) {
      this.setState(ExploreState.LOOKING_AROUND);
    }
  }
  
  /**
   * Pausing state
   */
  _updatePausing(dt, stateTime) {
    if (stateTime > this.options.pauseDuration * 0.5) {
      this.setState(ExploreState.LOOKING_AROUND);
    }
  }
  
  /**
   * Scan surroundings for objects and landmarks
   */
  scanSurroundings() {
    if (!this.player.model || !this.scene) return;
    
    const pos = this.player.model.position;
    const radius = this.options.observationRadius;
    this.nearbyObjects = [];
    
    // Scan scene for objects
    this.scene.traverse(obj => {
      if (!obj.isMesh && !obj.isGroup) return;
      if (obj === this.player.model) return;
      if (!obj.name || obj.name.startsWith("tile_") === false) return;
      
      // Calculate distance
      const objPos = new THREE.Vector3();
      obj.getWorldPosition(objPos);
      const distance = pos.distanceTo(objPos);
      
      if (distance > radius * 2) return;
      
      // Identify landmark type
      const name = obj.name.toLowerCase();
      let landmarkType = null;
      
      for (const [type, info] of Object.entries({
        fountain: { keywords: ["fountain"] },
        park: { keywords: ["grass", "trees", "forest"] },
        building: { keywords: ["building", "house", "office", "garage"] },
        road: { keywords: ["road", "pavement", "intersection", "corner", "split"] }
      })) {
        if (info.keywords.some(kw => name.includes(kw))) {
          landmarkType = type;
          break;
        }
      }
      
      if (!landmarkType) return;
      
      // Register as landmark if new
      if (!this.memory.landmarks.has(obj.name)) {
        const landmark = this.memory.discoverLandmark(
          obj.name,
          landmarkType,
          { x: objPos.x, y: objPos.y, z: objPos.z }
        );
        
        if (this.onLandmarkDiscovered) {
          this.onLandmarkDiscovered(landmark);
        }
      }
      
      // Track as nearby
      this.nearbyObjects.push({
        id: obj.name,
        type: landmarkType,
        distance,
        position: { x: objPos.x, y: objPos.y, z: objPos.z }
      });
    });
    
    // Sort by distance
    this.nearbyObjects.sort((a, b) => a.distance - b.distance);
  }
  
  /**
   * Get current exploration status
   */
  getStatus() {
    const pos = this.player.model?.position || { x: 0, y: 0, z: 0 };
    
    return {
      state: this.state,
      position: { x: pos.x, y: pos.y, z: pos.z },
      currentTarget: this.currentTarget,
      nearbyObjects: this.nearbyObjects.slice(0, 5),
      memory: this.memory.getSummary(),
      memoryDescription: this.memory.describe(pos)
    };
  }
  
  /**
   * Get natural language description of current activity
   */
  describeActivity() {
    const pos = this.player.model?.position || { x: 0, y: 0, z: 0 };
    
    switch (this.state) {
      case ExploreState.LOOKING_AROUND:
        return "I'm looking around to see what's nearby.";
        
      case ExploreState.CHOOSING_DIRECTION:
        return "I'm deciding where to go next.";
        
      case ExploreState.WALKING:
        if (this.currentTarget?.landmark) {
          return `I'm walking toward the ${this.currentTarget.landmark.name}.`;
        }
        return "I'm walking to explore a new area.";
        
      case ExploreState.OBSERVING_LANDMARK:
        if (this.currentTarget?.landmark) {
          return `I'm observing the ${this.currentTarget.landmark.name}.`;
        }
        return "I'm observing something interesting.";
        
      case ExploreState.PAUSING:
        return "I'm pausing to take in my surroundings.";
        
      default:
        return "I'm idle.";
    }
  }
  
  /**
   * Get memory reference for external use
   */
  getMemory() {
    return this.memory;
  }
}
