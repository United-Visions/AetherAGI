/**
 * BodyStateTracker.js
 * Tracks Aether's body state for proprioception and self-awareness
 * Provides real-time body awareness data to the AI backend
 */

import * as THREE from "three";

export class BodyStateTracker {
  constructor(playerRig, physicsWorld = null) {
    this.player = playerRig;
    this.physics = physicsWorld;
    
    // Body state history for velocity/acceleration calculation
    this.positionHistory = [];
    this.rotationHistory = [];
    this.maxHistoryLength = 10;
    
    // Proprioceptive state
    this.state = {
      // Position in world
      position: { x: 0, y: 0, z: 0 },
      rotation: 0,
      
      // Velocity and acceleration
      velocity: { x: 0, y: 0, z: 0 },
      angularVelocity: 0,
      speed: 0,
      
      // Body posture
      isGrounded: true,
      isMoving: false,
      isRunning: false,
      isFalling: false,
      isJumping: false,
      
      // Current animation/action
      currentAnimation: "idle",
      animationProgress: 0,
      
      // Limb positions (if skeleton available)
      limbs: {
        head: { x: 0, y: 0, z: 0 },
        leftHand: { x: 0, y: 0, z: 0 },
        rightHand: { x: 0, y: 0, z: 0 },
        leftFoot: { x: 0, y: 0, z: 0 },
        rightFoot: { x: 0, y: 0, z: 0 }
      },
      
      // Physical sensations
      sensations: {
        groundContact: true,
        nearbyObjects: [],
        touchingObjects: [],
        temperature: 20, // Simulated
        windDirection: { x: 0, y: 0, z: 0 }
      },
      
      // Energy/fatigue (simulated)
      energy: 100,
      fatigue: 0
    };
    
    // Bone references for limb tracking
    this.bones = null;
    
    console.log("🦴 Body state tracker initialized");
  }
  
  /**
   * Link skeleton bones after model loads
   */
  linkSkeleton(bones) {
    this.bones = bones;
  }
  
  /**
   * Update body state (call every frame)
   */
  update(dt) {
    if (!this.player.model) return this.state;
    
    const model = this.player.model;
    const now = Date.now();
    
    // Current position and rotation
    const currentPos = {
      x: model.position.x,
      y: model.position.y,
      z: model.position.z,
      time: now
    };
    const currentRot = model.rotation.y;
    
    // Calculate velocity from position history
    if (this.positionHistory.length > 0) {
      const prevPos = this.positionHistory[this.positionHistory.length - 1];
      const timeDelta = (now - prevPos.time) / 1000;
      
      if (timeDelta > 0) {
        this.state.velocity = {
          x: (currentPos.x - prevPos.x) / timeDelta,
          y: (currentPos.y - prevPos.y) / timeDelta,
          z: (currentPos.z - prevPos.z) / timeDelta
        };
        
        this.state.speed = Math.sqrt(
          this.state.velocity.x ** 2 +
          this.state.velocity.y ** 2 +
          this.state.velocity.z ** 2
        );
      }
    }
    
    // Calculate angular velocity
    if (this.rotationHistory.length > 0) {
      const prevRot = this.rotationHistory[this.rotationHistory.length - 1];
      const timeDelta = (now - prevRot.time) / 1000;
      
      if (timeDelta > 0) {
        let rotDiff = currentRot - prevRot.rotation;
        // Normalize to -PI to PI
        while (rotDiff > Math.PI) rotDiff -= Math.PI * 2;
        while (rotDiff < -Math.PI) rotDiff += Math.PI * 2;
        this.state.angularVelocity = rotDiff / timeDelta;
      }
    }
    
    // Update history
    this.positionHistory.push(currentPos);
    this.rotationHistory.push({ rotation: currentRot, time: now });
    
    // Trim history
    if (this.positionHistory.length > this.maxHistoryLength) {
      this.positionHistory.shift();
    }
    if (this.rotationHistory.length > this.maxHistoryLength) {
      this.rotationHistory.shift();
    }
    
    // Update position state
    this.state.position = { x: currentPos.x, y: currentPos.y, z: currentPos.z };
    this.state.rotation = currentRot;
    
    // Update movement state from player
    this.state.isMoving = this.player.isMoving || false;
    this.state.isRunning = this.player.isRunning || false;
    this.state.currentAnimation = this.player.currentAnimation || "idle";
    
    // Detect falling (negative Y velocity)
    this.state.isFalling = this.state.velocity.y < -1;
    this.state.isGrounded = currentPos.y <= 0.1 && !this.state.isFalling;
    
    // Update limb positions from skeleton
    if (this.bones || this.player.bones) {
      this._updateLimbPositions();
    }
    
    // Query nearby objects from physics
    if (this.physics) {
      this.state.sensations.nearbyObjects = this.physics.queryNearby(currentPos, 3);
    }
    
    // Simulate energy/fatigue
    if (this.state.isRunning) {
      this.state.energy = Math.max(0, this.state.energy - dt * 5);
      this.state.fatigue = Math.min(100, this.state.fatigue + dt * 3);
    } else if (this.state.isMoving) {
      this.state.energy = Math.max(0, this.state.energy - dt * 1);
      this.state.fatigue = Math.min(100, this.state.fatigue + dt * 0.5);
    } else {
      // Recover while idle
      this.state.energy = Math.min(100, this.state.energy + dt * 2);
      this.state.fatigue = Math.max(0, this.state.fatigue - dt * 1);
    }
    
    return this.state;
  }
  
  /**
   * Update limb positions from skeleton bones
   */
  _updateLimbPositions() {
    const bones = this.bones || this.player.bones || {};
    const worldPos = new THREE.Vector3();
    
    const getBoneWorldPos = (bone) => {
      if (!bone) return { x: 0, y: 0, z: 0 };
      bone.getWorldPosition(worldPos);
      return { x: worldPos.x, y: worldPos.y, z: worldPos.z };
    };
    
    if (bones.headBone) {
      this.state.limbs.head = getBoneWorldPos(bones.headBone);
    }
    if (bones.leftArm) {
      this.state.limbs.leftHand = getBoneWorldPos(bones.leftArm);
    }
    if (bones.rightArm) {
      this.state.limbs.rightHand = getBoneWorldPos(bones.rightArm);
    }
    if (bones.leftLeg) {
      this.state.limbs.leftFoot = getBoneWorldPos(bones.leftLeg);
    }
    if (bones.rightLeg) {
      this.state.limbs.rightFoot = getBoneWorldPos(bones.rightLeg);
    }
  }
  
  /**
   * Get a natural language description of current body state
   */
  describeState() {
    const parts = [];
    
    // Position
    parts.push(`I am at position (${this.state.position.x.toFixed(1)}, ${this.state.position.y.toFixed(1)}, ${this.state.position.z.toFixed(1)})`);
    
    // Movement
    if (this.state.isRunning) {
      parts.push(`running at ${this.state.speed.toFixed(1)} m/s`);
    } else if (this.state.isMoving) {
      parts.push(`walking at ${this.state.speed.toFixed(1)} m/s`);
    } else {
      parts.push("standing still");
    }
    
    // Grounded
    if (!this.state.isGrounded) {
      if (this.state.isFalling) {
        parts.push("and falling");
      } else {
        parts.push("in the air");
      }
    }
    
    // Animation
    if (this.state.currentAnimation !== "idle") {
      parts.push(`performing ${this.state.currentAnimation} animation`);
    }
    
    // Energy
    if (this.state.energy < 30) {
      parts.push("feeling tired");
    } else if (this.state.energy > 90) {
      parts.push("feeling energized");
    }
    
    // Nearby objects
    if (this.state.sensations.nearbyObjects.length > 0) {
      const nearby = this.state.sensations.nearbyObjects.slice(0, 3);
      const names = nearby.map(o => o.name).filter(n => n !== "unknown");
      if (names.length > 0) {
        parts.push(`near ${names.join(", ")}`);
      }
    }
    
    return parts.join(", ") + ".";
  }
  
  /**
   * Serialize full state for backend
   */
  serialize() {
    return {
      ...this.state,
      description: this.describeState(),
      timestamp: Date.now()
    };
  }
}
