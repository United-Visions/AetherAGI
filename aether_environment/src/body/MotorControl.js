/**
 * MotorControl.js
 * 
 * Real humanoid-style motor control for Aether.
 * Gives direct control over joints/bones like a real robot:
 * - Inverse Kinematics (IK) for reaching targets
 * - Direct joint angle control
 * - Finger articulation
 * - Head/eye tracking
 * - Procedural balance and posture
 * 
 * This replaces canned animations with true articulated control.
 */

import * as THREE from "three";

// Joint limits (in radians) - based on human anatomy
const JOINT_LIMITS = {
  // Neck/Head
  neck: { pitch: [-0.7, 0.5], yaw: [-1.2, 1.2], roll: [-0.4, 0.4] },
  head: { pitch: [-0.3, 0.3], yaw: [-0.5, 0.5], roll: [-0.2, 0.2] },
  
  // Spine
  spine: { pitch: [-0.3, 0.4], yaw: [-0.5, 0.5], roll: [-0.3, 0.3] },
  chest: { pitch: [-0.2, 0.3], yaw: [-0.4, 0.4], roll: [-0.2, 0.2] },
  
  // Arms
  shoulder: { pitch: [-1.0, 3.14], yaw: [-1.5, 0.3], roll: [-1.5, 1.5] },
  upperArm: { pitch: [-2.6, 0.5], yaw: [-1.5, 1.5], roll: [-1.5, 1.5] },
  elbow: { pitch: [0, 2.5], yaw: [0, 0], roll: [-1.5, 1.5] },
  wrist: { pitch: [-1.0, 1.0], yaw: [-0.5, 0.5], roll: [-1.5, 1.5] },
  
  // Fingers (simplified - just curl amount 0-1)
  finger: { curl: [0, 1], spread: [-0.3, 0.3] },
  
  // Legs
  hip: { pitch: [-0.5, 2.0], yaw: [-0.5, 0.8], roll: [-0.5, 0.5] },
  knee: { pitch: [0, 2.5], yaw: [0, 0], roll: [0, 0] },
  ankle: { pitch: [-0.7, 0.7], yaw: [-0.3, 0.3], roll: [-0.3, 0.3] }
};

// Bone name patterns for different rigs (Mixamo, Blender, etc.)
const BONE_PATTERNS = {
  hips: ["hips", "pelvis", "root"],
  spine: ["spine", "spine1", "spine01"],
  chest: ["spine2", "spine02", "chest", "spine1"],
  neck: ["neck"],
  head: ["head"],
  
  leftShoulder: ["leftshoulder", "shoulder.l", "l_shoulder"],
  leftUpperArm: ["leftarm", "upperarm.l", "l_upperarm", "leftupperarm"],
  leftForeArm: ["leftforearm", "forearm.l", "l_forearm", "lowerarm.l"],
  leftHand: ["lefthand", "hand.l", "l_hand"],
  
  rightShoulder: ["rightshoulder", "shoulder.r", "r_shoulder"],
  rightUpperArm: ["rightarm", "upperarm.r", "r_upperarm", "rightupperarm"],
  rightForeArm: ["rightforearm", "forearm.r", "r_forearm", "lowerarm.r"],
  rightHand: ["righthand", "hand.r", "r_hand"],
  
  leftUpLeg: ["leftupleg", "thigh.l", "l_thigh", "leftthigh"],
  leftLeg: ["leftleg", "shin.l", "l_shin", "leftlowerleg", "calf.l"],
  leftFoot: ["leftfoot", "foot.l", "l_foot"],
  leftToe: ["lefttoebase", "toe.l", "l_toe"],
  
  rightUpLeg: ["rightupleg", "thigh.r", "r_thigh", "rightthigh"],
  rightLeg: ["rightleg", "shin.r", "r_shin", "rightlowerleg", "calf.r"],
  rightFoot: ["rightfoot", "foot.r", "r_foot"],
  rightToe: ["righttoebase", "toe.r", "r_toe"],
  
  // Fingers (Mixamo naming)
  leftThumb1: ["lefthandthumb1", "thumb01.l"],
  leftThumb2: ["lefthandthumb2", "thumb02.l"],
  leftThumb3: ["lefthandthumb3", "thumb03.l"],
  leftIndex1: ["lefthandindex1", "index01.l"],
  leftIndex2: ["lefthandindex2", "index02.l"],
  leftIndex3: ["lefthandindex3", "index03.l"],
  leftMiddle1: ["lefthandmiddle1", "middle01.l"],
  leftMiddle2: ["lefthandmiddle2", "middle02.l"],
  leftMiddle3: ["lefthandmiddle3", "middle03.l"],
  leftRing1: ["lefthandring1", "ring01.l"],
  leftRing2: ["lefthandring2", "ring02.l"],
  leftRing3: ["lefthandring3", "ring03.l"],
  leftPinky1: ["lefthandpinky1", "pinky01.l"],
  leftPinky2: ["lefthandpinky2", "pinky02.l"],
  leftPinky3: ["lefthandpinky3", "pinky03.l"],
  
  rightThumb1: ["righthandthumb1", "thumb01.r"],
  rightThumb2: ["righthandthumb2", "thumb02.r"],
  rightThumb3: ["righthandthumb3", "thumb03.r"],
  rightIndex1: ["righthandindex1", "index01.r"],
  rightIndex2: ["righthandindex2", "index02.r"],
  rightIndex3: ["righthandindex3", "index03.r"],
  rightMiddle1: ["righthandmiddle1", "middle01.r"],
  rightMiddle2: ["righthandmiddle2", "middle02.r"],
  rightMiddle3: ["righthandmiddle3", "middle03.r"],
  rightRing1: ["righthandring1", "ring01.r"],
  rightRing2: ["righthandring2", "ring02.r"],
  rightRing3: ["righthandring3", "ring03.r"],
  rightPinky1: ["righthandpinky1", "pinky01.r"],
  rightPinky2: ["righthandpinky2", "pinky02.r"],
  rightPinky3: ["righthandpinky3", "pinky03.r"]
};

export class MotorControl {
  constructor(model) {
    this.model = model;
    this.skeleton = null;
    this.bones = {};
    this.restPoses = {}; // Store initial bone rotations
    
    // Current joint states (target angles)
    this.jointStates = {};
    
    // Smooth interpolation
    this.lerpSpeed = 5.0; // How fast joints move to targets
    
    // IK targets
    this.ikTargets = {
      leftHand: null,
      rightHand: null,
      leftFoot: null,
      rightFoot: null,
      head: null // Look-at target
    };
    
    // Motor control enabled
    this.enabled = false;
    
    // Initialize
    this._findBones();
    this._storeRestPoses();
    
    console.log("🦾 Motor control initialized with", Object.keys(this.bones).length, "bones");
  }
  
  /**
   * Find and map all bones in the skeleton
   */
  _findBones() {
    if (!this.model) return;
    
    const allBones = [];
    
    this.model.traverse(child => {
      if (child.isBone) {
        allBones.push(child);
        
        // Try to match to our bone patterns
        const nameLower = child.name.toLowerCase().replace(/[^a-z0-9]/g, '');
        
        for (const [boneName, patterns] of Object.entries(BONE_PATTERNS)) {
          for (const pattern of patterns) {
            const patternClean = pattern.replace(/[^a-z0-9]/g, '');
            if (nameLower.includes(patternClean) || nameLower === patternClean) {
              if (!this.bones[boneName]) {
                this.bones[boneName] = child;
                console.log(`  Mapped ${boneName} → ${child.name}`);
              }
              break;
            }
          }
        }
      }
      
      // Find skeleton
      if (child.isSkinnedMesh && child.skeleton) {
        this.skeleton = child.skeleton;
      }
    });
    
    console.log("🦴 Found bones:", Object.keys(this.bones).join(", "));
    console.log("🦴 Total bones in model:", allBones.length);
  }
  
  /**
   * Store rest poses for all bones
   */
  _storeRestPoses() {
    for (const [name, bone] of Object.entries(this.bones)) {
      this.restPoses[name] = {
        position: bone.position.clone(),
        quaternion: bone.quaternion.clone(),
        rotation: bone.rotation.clone()
      };
      
      // Initialize joint state
      this.jointStates[name] = {
        current: { x: 0, y: 0, z: 0 },
        target: { x: 0, y: 0, z: 0 }
      };
    }
  }
  
  /**
   * Enable motor control (disables animation mixer)
   */
  enable() {
    this.enabled = true;
    console.log("🦾 Motor control ENABLED - Aether has direct joint control");
  }
  
  /**
   * Disable motor control (re-enables animations)
   */
  disable() {
    this.enabled = false;
    this._resetToRestPose();
    console.log("🦾 Motor control disabled - returning to animations");
  }
  
  /**
   * Reset all bones to rest pose
   */
  _resetToRestPose() {
    for (const [name, bone] of Object.entries(this.bones)) {
      const rest = this.restPoses[name];
      if (rest) {
        bone.quaternion.copy(rest.quaternion);
      }
    }
  }
  
  // ═══════════════════════════════════════════════════════════════════════════
  // DIRECT JOINT CONTROL
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Set target angle for a joint
   * @param {string} boneName - Name of the bone (e.g., "leftUpperArm")
   * @param {object} angles - { pitch, yaw, roll } in radians, relative to rest pose
   */
  setJointAngle(boneName, angles) {
    if (!this.jointStates[boneName]) return;
    
    const state = this.jointStates[boneName];
    state.target.x = angles.pitch ?? state.target.x;
    state.target.y = angles.yaw ?? state.target.y;
    state.target.z = angles.roll ?? state.target.z;
  }
  
  /**
   * Raise left arm (convenience method)
   * @param {number} amount - 0 (down) to 1 (fully raised)
   */
  raiseLeftArm(amount) {
    const angle = THREE.MathUtils.lerp(0, Math.PI * 0.75, amount);
    this.setJointAngle("leftUpperArm", { pitch: -angle, roll: 0.2 });
    this.setJointAngle("leftForeArm", { pitch: amount * 0.3 });
  }
  
  /**
   * Raise right arm
   */
  raiseRightArm(amount) {
    const angle = THREE.MathUtils.lerp(0, Math.PI * 0.75, amount);
    this.setJointAngle("rightUpperArm", { pitch: -angle, roll: -0.2 });
    this.setJointAngle("rightForeArm", { pitch: amount * 0.3 });
  }
  
  /**
   * Wave hand
   */
  wave(hand = "right", wavePhase = 0) {
    const isRight = hand === "right";
    const armName = isRight ? "rightUpperArm" : "leftUpperArm";
    const forearmName = isRight ? "rightForeArm" : "leftForeArm";
    const handName = isRight ? "rightHand" : "leftHand";
    
    // Raise arm
    this.setJointAngle(armName, { 
      pitch: -Math.PI * 0.6, 
      yaw: isRight ? -0.3 : 0.3,
      roll: isRight ? -0.5 : 0.5 
    });
    
    // Bend elbow
    this.setJointAngle(forearmName, { pitch: Math.PI * 0.4 });
    
    // Wave the hand
    const waveAmount = Math.sin(wavePhase) * 0.4;
    this.setJointAngle(handName, { roll: waveAmount });
  }
  
  /**
   * Point at a world position
   */
  pointAt(worldPosition, hand = "right") {
    if (!this.model) return;
    
    const isRight = hand === "right";
    const shoulderName = isRight ? "rightShoulder" : "leftShoulder";
    const upperArmName = isRight ? "rightUpperArm" : "leftUpperArm";
    const forearmName = isRight ? "rightForeArm" : "leftForeArm";
    
    const shoulder = this.bones[shoulderName] || this.bones[upperArmName];
    if (!shoulder) return;
    
    // Get shoulder world position
    const shoulderWorld = new THREE.Vector3();
    shoulder.getWorldPosition(shoulderWorld);
    
    // Calculate direction to target
    const direction = new THREE.Vector3()
      .subVectors(worldPosition, shoulderWorld)
      .normalize();
    
    // Convert to angles
    const pitch = -Math.asin(direction.y);
    const yaw = Math.atan2(direction.x, direction.z);
    
    // Adjust for body rotation
    const bodyRotation = this.model.rotation.y;
    const relativeYaw = yaw - bodyRotation;
    
    this.setJointAngle(upperArmName, {
      pitch: pitch - Math.PI * 0.5,
      yaw: isRight ? -relativeYaw : relativeYaw,
      roll: 0
    });
    
    // Straighten forearm for pointing
    this.setJointAngle(forearmName, { pitch: 0.1 });
  }
  
  /**
   * Look at a world position (turn head)
   */
  lookAt(worldPosition) {
    if (!this.model) return;
    
    const head = this.bones.head;
    const neck = this.bones.neck;
    if (!head && !neck) return;
    
    // Get head world position
    const headBone = head || neck;
    const headWorld = new THREE.Vector3();
    headBone.getWorldPosition(headWorld);
    
    // Calculate direction to target
    const direction = new THREE.Vector3()
      .subVectors(worldPosition, headWorld)
      .normalize();
    
    // Calculate angles
    const pitch = Math.asin(-direction.y) * 0.7; // Limit pitch
    const yaw = Math.atan2(direction.x, direction.z);
    
    // Adjust for body rotation
    const bodyRotation = this.model.rotation.y;
    let relativeYaw = yaw - bodyRotation;
    
    // Clamp yaw
    relativeYaw = THREE.MathUtils.clamp(relativeYaw, -1.0, 1.0);
    
    // Distribute between neck and head
    if (neck) {
      this.setJointAngle("neck", { 
        pitch: pitch * 0.4, 
        yaw: relativeYaw * 0.6 
      });
    }
    if (head) {
      this.setJointAngle("head", { 
        pitch: pitch * 0.6, 
        yaw: relativeYaw * 0.4 
      });
    }
  }
  
  // ═══════════════════════════════════════════════════════════════════════════
  // FINGER CONTROL
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Set finger curl (0 = open, 1 = closed fist)
   */
  setFingerCurl(hand, finger, curl) {
    const prefix = hand === "right" ? "right" : "left";
    const fingerNames = {
      thumb: [`${prefix}Thumb1`, `${prefix}Thumb2`, `${prefix}Thumb3`],
      index: [`${prefix}Index1`, `${prefix}Index2`, `${prefix}Index3`],
      middle: [`${prefix}Middle1`, `${prefix}Middle2`, `${prefix}Middle3`],
      ring: [`${prefix}Ring1`, `${prefix}Ring2`, `${prefix}Ring3`],
      pinky: [`${prefix}Pinky1`, `${prefix}Pinky2`, `${prefix}Pinky3`]
    };
    
    const segments = fingerNames[finger];
    if (!segments) return;
    
    // Curl increases down the finger
    const curlAngles = [curl * 0.6, curl * 0.8, curl * 0.9];
    
    segments.forEach((segName, i) => {
      if (this.bones[segName]) {
        this.setJointAngle(segName, { pitch: curlAngles[i] });
      }
    });
  }
  
  /**
   * Set all fingers at once
   */
  setHandPose(hand, pose) {
    const fingers = ["thumb", "index", "middle", "ring", "pinky"];
    
    switch (pose) {
      case "open":
        fingers.forEach(f => this.setFingerCurl(hand, f, 0));
        break;
      case "fist":
        fingers.forEach(f => this.setFingerCurl(hand, f, 1));
        break;
      case "point":
        this.setFingerCurl(hand, "thumb", 0.5);
        this.setFingerCurl(hand, "index", 0);
        this.setFingerCurl(hand, "middle", 1);
        this.setFingerCurl(hand, "ring", 1);
        this.setFingerCurl(hand, "pinky", 1);
        break;
      case "peace":
        this.setFingerCurl(hand, "thumb", 0.5);
        this.setFingerCurl(hand, "index", 0);
        this.setFingerCurl(hand, "middle", 0);
        this.setFingerCurl(hand, "ring", 1);
        this.setFingerCurl(hand, "pinky", 1);
        break;
      case "thumbsUp":
        this.setFingerCurl(hand, "thumb", 0);
        this.setFingerCurl(hand, "index", 1);
        this.setFingerCurl(hand, "middle", 1);
        this.setFingerCurl(hand, "ring", 1);
        this.setFingerCurl(hand, "pinky", 1);
        break;
      case "grab":
        fingers.forEach(f => this.setFingerCurl(hand, f, 0.7));
        break;
    }
  }
  
  // ═══════════════════════════════════════════════════════════════════════════
  // INVERSE KINEMATICS (Simple 2-bone IK)
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Reach hand to world position using IK
   */
  reachTo(worldPosition, hand = "right") {
    this.ikTargets[hand === "right" ? "rightHand" : "leftHand"] = worldPosition.clone();
  }
  
  /**
   * Clear IK target
   */
  clearReach(hand) {
    this.ikTargets[hand === "right" ? "rightHand" : "leftHand"] = null;
  }
  
  /**
   * Solve 2-bone IK for arm
   */
  _solveArmIK(hand) {
    const isRight = hand === "right";
    const target = this.ikTargets[isRight ? "rightHand" : "leftHand"];
    if (!target) return;
    
    const upperArm = this.bones[isRight ? "rightUpperArm" : "leftUpperArm"];
    const forearm = this.bones[isRight ? "rightForeArm" : "leftForeArm"];
    const handBone = this.bones[isRight ? "rightHand" : "leftHand"];
    
    if (!upperArm || !forearm) return;
    
    // Get bone lengths (approximate)
    const upperArmLength = 0.3;
    const forearmLength = 0.25;
    
    // Get shoulder position
    const shoulderPos = new THREE.Vector3();
    upperArm.getWorldPosition(shoulderPos);
    
    // Direction and distance to target
    const toTarget = new THREE.Vector3().subVectors(target, shoulderPos);
    const distance = toTarget.length();
    
    // Clamp distance to arm reach
    const maxReach = upperArmLength + forearmLength;
    const minReach = Math.abs(upperArmLength - forearmLength);
    const clampedDist = THREE.MathUtils.clamp(distance, minReach + 0.01, maxReach - 0.01);
    
    // Law of cosines for elbow angle
    const cosElbow = (upperArmLength * upperArmLength + forearmLength * forearmLength - clampedDist * clampedDist) 
                    / (2 * upperArmLength * forearmLength);
    const elbowAngle = Math.PI - Math.acos(THREE.MathUtils.clamp(cosElbow, -1, 1));
    
    // Shoulder angle
    const cosShoulder = (upperArmLength * upperArmLength + clampedDist * clampedDist - forearmLength * forearmLength)
                       / (2 * upperArmLength * clampedDist);
    const shoulderAngle = Math.acos(THREE.MathUtils.clamp(cosShoulder, -1, 1));
    
    // Direction angles
    toTarget.normalize();
    const pitch = -Math.asin(toTarget.y) - shoulderAngle;
    const yaw = Math.atan2(toTarget.x, toTarget.z) - this.model.rotation.y;
    
    // Apply to joints
    this.setJointAngle(isRight ? "rightUpperArm" : "leftUpperArm", {
      pitch: pitch,
      yaw: isRight ? -yaw * 0.5 : yaw * 0.5,
      roll: 0
    });
    
    this.setJointAngle(isRight ? "rightForeArm" : "leftForeArm", {
      pitch: elbowAngle
    });
  }
  
  // ═══════════════════════════════════════════════════════════════════════════
  // UPDATE LOOP
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Update motor control (call every frame)
   */
  update(deltaTime) {
    if (!this.enabled) return;
    
    // Solve IK for any active targets
    if (this.ikTargets.rightHand) this._solveArmIK("right");
    if (this.ikTargets.leftHand) this._solveArmIK("left");
    
    // Interpolate all joints toward their targets
    for (const [boneName, state] of Object.entries(this.jointStates)) {
      const bone = this.bones[boneName];
      if (!bone) continue;
      
      const rest = this.restPoses[boneName];
      if (!rest) continue;
      
      // Lerp current toward target
      const t = Math.min(1, deltaTime * this.lerpSpeed);
      state.current.x += (state.target.x - state.current.x) * t;
      state.current.y += (state.target.y - state.current.y) * t;
      state.current.z += (state.target.z - state.current.z) * t;
      
      // Apply rotation: rest pose + current offset
      const offsetQuat = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(state.current.x, state.current.y, state.current.z, "YXZ")
      );
      
      bone.quaternion.copy(rest.quaternion).multiply(offsetQuat);
    }
  }
  
  // ═══════════════════════════════════════════════════════════════════════════
  // PROPRIOCEPTION - Get current body state
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Get all joint angles (proprioception)
   */
  getJointAngles() {
    const angles = {};
    for (const [name, state] of Object.entries(this.jointStates)) {
      angles[name] = { ...state.current };
    }
    return angles;
  }
  
  /**
   * Get end effector positions (hands, feet, head)
   */
  getEndEffectorPositions() {
    const positions = {};
    
    const effectors = ["leftHand", "rightHand", "leftFoot", "rightFoot", "head"];
    for (const name of effectors) {
      const bone = this.bones[name];
      if (bone) {
        const pos = new THREE.Vector3();
        bone.getWorldPosition(pos);
        positions[name] = { x: pos.x, y: pos.y, z: pos.z };
      }
    }
    
    return positions;
  }
  
  /**
   * Describe current pose in natural language
   */
  describePose() {
    const parts = [];
    
    // Check arm positions
    const leftArm = this.jointStates.leftUpperArm?.current;
    const rightArm = this.jointStates.rightUpperArm?.current;
    
    if (leftArm && leftArm.x < -0.5) parts.push("left arm raised");
    if (rightArm && rightArm.x < -0.5) parts.push("right arm raised");
    
    // Check head direction
    const head = this.jointStates.head?.current;
    const neck = this.jointStates.neck?.current;
    if (head || neck) {
      const yaw = (head?.y || 0) + (neck?.y || 0);
      if (yaw < -0.3) parts.push("looking right");
      else if (yaw > 0.3) parts.push("looking left");
      
      const pitch = (head?.x || 0) + (neck?.x || 0);
      if (pitch < -0.2) parts.push("looking up");
      else if (pitch > 0.2) parts.push("looking down");
    }
    
    return parts.length > 0 ? parts.join(", ") : "neutral pose";
  }
}
