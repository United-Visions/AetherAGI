import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { clone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { MotorControl } from "../body/MotorControl.js";

const loader = new GLTFLoader();

export class PlayerRig {
  constructor(scene, registry, options = {}) {
    this.scene = scene;
    this.registry = registry;
    this.options = {
      characterAsset: "/assets/characters/aether.glb",
      animationLibrary: "/assets/animations/universal_animations.glb",
      scale: 0.015,
      initialPosition: { x: 0, y: 0, z: 0 },
      initialRotation: 0, // Radians - 0 faces -Z, Math.PI faces +Z
      moveSpeed: 1.5,
      runSpeed: 3.0,
      turnSpeed: 5.0,
      ...options
    };

    this.mixer = null;
    this.actions = {};
    this.activeAction = null;
    this.currentAnimation = "idle";
    this.model = null;
    this.velocity = new THREE.Vector3();
    this.targetRotation = 0;
    this.tmpQuat = new THREE.Quaternion();
    
    // Motor control system (real humanoid-style joint control)
    this.motorControl = null;
    this.useMotorControl = false; // Toggle between animations and motor control
    
    // Input state
    this.input = {
      forward: false,
      backward: false,
      left: false,
      right: false,
      run: false,
      wave: false,
      jump: false
    };
    
    // Movement state
    this.isMoving = false;
    this.isRunning = false;
    
    // AI-controlled walking state
    this.aiControlled = false;
    this.walkTarget = null;

    this._load();
    this._setupControls();
  }

  async _load() {
    try {
      // Load character model
      const character = await loader.loadAsync(this.options.characterAsset);
      
      this.model = clone(character.scene);
      this.model.scale.setScalar(this.options.scale);
      
      // Apply initial position
      const pos = this.options.initialPosition;
      this.model.position.set(pos.x ?? 0, pos.y ?? 0, pos.z ?? 0);
      
      // Apply initial rotation
      this.model.rotation.y = this.options.initialRotation ?? 0;
      this.targetRotation = this.options.initialRotation ?? 0;
      
      this.model.traverse(child => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });
      this.scene.add(this.model);

      this.mixer = new THREE.AnimationMixer(this.model);
      
      // Check if character has embedded animations
      const characterAnims = character.animations || [];
      console.log("Character embedded animations:", characterAnims.map(c => c.name));
      
      // Find skeleton bones for procedural animations
      let rootBone = null;
      let spineBone = null;
      let headBone = null;
      let leftArm = null;
      let rightArm = null;
      let leftLeg = null;
      let rightLeg = null;
      
      this.model.traverse(child => {
        const name = child.name.toLowerCase();
        if (child.isBone) {
          // Common bone name patterns (Mixamo, Blender, etc.)
          if (name.includes("hips") || name.includes("pelvis") || name.includes("root")) {
            rootBone = child;
          }
          if (name.includes("spine") && !spineBone) {
            spineBone = child;
          }
          if (name.includes("head") && !headBone) {
            headBone = child;
          }
          if ((name.includes("arm") || name.includes("shoulder")) && name.includes("left") && !leftArm) {
            leftArm = child;
          }
          if ((name.includes("arm") || name.includes("shoulder")) && name.includes("right") && !rightArm) {
            rightArm = child;
          }
          if ((name.includes("leg") || name.includes("thigh") || name.includes("upleg")) && name.includes("left") && !leftLeg) {
            leftLeg = child;
          }
          if ((name.includes("leg") || name.includes("thigh") || name.includes("upleg")) && name.includes("right") && !rightLeg) {
            rightLeg = child;
          }
        }
      });
      
      console.log("Found bones - Root:", rootBone?.name, "Spine:", spineBone?.name, 
                  "Head:", headBone?.name, "Arms:", leftArm?.name, rightArm?.name);
      
      this.bones = { rootBone, spineBone, headBone, leftArm, rightArm, leftLeg, rightLeg };
      
      // Try to use embedded animations first
      let animationsWorked = false;
      if (characterAnims.length > 0) {
        characterAnims.forEach(clip => {
          const name = clip.name.toLowerCase();
          const action = this.mixer.clipAction(clip);
          
          if (name.includes("idle") || name.includes("breathing")) {
            this.actions.idle = action;
            animationsWorked = true;
          }
          if (name.includes("walk") && !name.includes("run")) {
            this.actions.walk = action;
          }
          if (name.includes("run") || name.includes("sprint")) {
            this.actions.run = action;
          }
        });
      }
      
      // If no embedded animations, create procedural ones
      if (!animationsWorked || !this.actions.idle) {
        console.log("Creating procedural animations for character");
        this._createProceduralAnimations();
      }
      
      console.log("Available actions:", Object.keys(this.actions));
      this._play("idle");
      
      // Initialize motor control system for humanoid-style joint control
      this.motorControl = new MotorControl(this.model);
      console.log("🦾 Motor control system ready - Aether can control individual joints");
      
    } catch (error) {
      console.warn("Fallback to capsule player - asset missing?", error);
      const capsule = new THREE.Mesh(
        new THREE.CapsuleGeometry(0.4, 1.5, 8, 16),
        new THREE.MeshStandardMaterial({ color: 0x4bc1ee })
      );
      capsule.castShadow = true;
      this.model = capsule;
      this.model.scale.setScalar(this.options.scale ?? 1);
      
      const pos = this.options.initialPosition;
      this.model.position.set(pos.x ?? 0, pos.y ?? 0, pos.z ?? 0);
      
      this.scene.add(this.model);
    }
  }
  
  _createProceduralAnimations() {
    const { rootBone, spineBone, headBone, leftArm, rightArm, leftLeg, rightLeg } = this.bones || {};
    
    // ═══════════════════════════════════════════════════════════════════════
    // IDLE ANIMATION - Gentle breathing motion
    // ═══════════════════════════════════════════════════════════════════════
    const idleTracks = [];
    const idleDuration = 3.0;
    
    if (rootBone) {
      // Subtle up/down breathing
      const baseY = rootBone.position.y;
      idleTracks.push(new THREE.VectorKeyframeTrack(
        `${rootBone.name}.position`,
        [0, 1.5, 3.0],
        [
          rootBone.position.x, baseY, rootBone.position.z,
          rootBone.position.x, baseY + 0.01, rootBone.position.z,
          rootBone.position.x, baseY, rootBone.position.z
        ]
      ));
    }
    
    if (spineBone) {
      // Slight spine sway
      const baseRot = spineBone.quaternion.clone();
      const swayQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.02, 0, 0.01));
      const swayedQuat = baseRot.clone().multiply(swayQuat);
      
      idleTracks.push(new THREE.QuaternionKeyframeTrack(
        `${spineBone.name}.quaternion`,
        [0, 1.5, 3.0],
        [
          baseRot.x, baseRot.y, baseRot.z, baseRot.w,
          swayedQuat.x, swayedQuat.y, swayedQuat.z, swayedQuat.w,
          baseRot.x, baseRot.y, baseRot.z, baseRot.w
        ]
      ));
    }
    
    if (headBone) {
      // Subtle head movement (looking around slightly)
      const baseRot = headBone.quaternion.clone();
      const lookLeft = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0.05, 0));
      const lookRight = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -0.03, 0.02));
      const leftQuat = baseRot.clone().multiply(lookLeft);
      const rightQuat = baseRot.clone().multiply(lookRight);
      
      idleTracks.push(new THREE.QuaternionKeyframeTrack(
        `${headBone.name}.quaternion`,
        [0, 1.0, 2.0, 3.0],
        [
          baseRot.x, baseRot.y, baseRot.z, baseRot.w,
          leftQuat.x, leftQuat.y, leftQuat.z, leftQuat.w,
          rightQuat.x, rightQuat.y, rightQuat.z, rightQuat.w,
          baseRot.x, baseRot.y, baseRot.z, baseRot.w
        ]
      ));
    }
    
    if (idleTracks.length > 0) {
      const idleClip = new THREE.AnimationClip('ProceduralIdle', idleDuration, idleTracks);
      this.actions.idle = this.mixer.clipAction(idleClip);
    } else {
      // Fallback: just bob the whole model
      const bobTrack = new THREE.VectorKeyframeTrack(
        '.position',
        [0, 1.5, 3.0],
        [
          this.model.position.x, this.model.position.y, this.model.position.z,
          this.model.position.x, this.model.position.y + 0.02, this.model.position.z,
          this.model.position.x, this.model.position.y, this.model.position.z
        ]
      );
      const idleClip = new THREE.AnimationClip('ProceduralIdle', idleDuration, [bobTrack]);
      this.actions.idle = this.mixer.clipAction(idleClip);
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // WALK ANIMATION - Arm and leg swing
    // ═══════════════════════════════════════════════════════════════════════
    const walkTracks = [];
    const walkDuration = 1.0;
    
    if (rootBone) {
      // Slight bounce while walking
      const baseY = rootBone.position.y;
      walkTracks.push(new THREE.VectorKeyframeTrack(
        `${rootBone.name}.position`,
        [0, 0.25, 0.5, 0.75, 1.0],
        [
          rootBone.position.x, baseY, rootBone.position.z,
          rootBone.position.x, baseY + 0.015, rootBone.position.z,
          rootBone.position.x, baseY, rootBone.position.z,
          rootBone.position.x, baseY + 0.015, rootBone.position.z,
          rootBone.position.x, baseY, rootBone.position.z
        ]
      ));
    }
    
    // Arm swing
    if (leftArm && rightArm) {
      const leftBase = leftArm.quaternion.clone();
      const rightBase = rightArm.quaternion.clone();
      const swingFwd = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.3, 0, 0));
      const swingBack = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.3, 0, 0));
      
      walkTracks.push(new THREE.QuaternionKeyframeTrack(
        `${leftArm.name}.quaternion`,
        [0, 0.5, 1.0],
        [
          ...leftBase.clone().multiply(swingFwd).toArray(),
          ...leftBase.clone().multiply(swingBack).toArray(),
          ...leftBase.clone().multiply(swingFwd).toArray()
        ]
      ));
      
      walkTracks.push(new THREE.QuaternionKeyframeTrack(
        `${rightArm.name}.quaternion`,
        [0, 0.5, 1.0],
        [
          ...rightBase.clone().multiply(swingBack).toArray(),
          ...rightBase.clone().multiply(swingFwd).toArray(),
          ...rightBase.clone().multiply(swingBack).toArray()
        ]
      ));
    }
    
    // Leg swing
    if (leftLeg && rightLeg) {
      const leftBase = leftLeg.quaternion.clone();
      const rightBase = rightLeg.quaternion.clone();
      const legFwd = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.4, 0, 0));
      const legBack = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.3, 0, 0));
      
      walkTracks.push(new THREE.QuaternionKeyframeTrack(
        `${leftLeg.name}.quaternion`,
        [0, 0.5, 1.0],
        [
          ...leftBase.clone().multiply(legFwd).toArray(),
          ...leftBase.clone().multiply(legBack).toArray(),
          ...leftBase.clone().multiply(legFwd).toArray()
        ]
      ));
      
      walkTracks.push(new THREE.QuaternionKeyframeTrack(
        `${rightLeg.name}.quaternion`,
        [0, 0.5, 1.0],
        [
          ...rightBase.clone().multiply(legBack).toArray(),
          ...rightBase.clone().multiply(legFwd).toArray(),
          ...rightBase.clone().multiply(legBack).toArray()
        ]
      ));
    }
    
    if (walkTracks.length > 0) {
      const walkClip = new THREE.AnimationClip('ProceduralWalk', walkDuration, walkTracks);
      this.actions.walk = this.mixer.clipAction(walkClip);
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // RUN ANIMATION - Faster, more exaggerated walk
    // ═══════════════════════════════════════════════════════════════════════
    const runTracks = [];
    const runDuration = 0.6;
    
    if (rootBone) {
      const baseY = rootBone.position.y;
      runTracks.push(new THREE.VectorKeyframeTrack(
        `${rootBone.name}.position`,
        [0, 0.15, 0.3, 0.45, 0.6],
        [
          rootBone.position.x, baseY, rootBone.position.z,
          rootBone.position.x, baseY + 0.03, rootBone.position.z,
          rootBone.position.x, baseY, rootBone.position.z,
          rootBone.position.x, baseY + 0.03, rootBone.position.z,
          rootBone.position.x, baseY, rootBone.position.z
        ]
      ));
    }
    
    if (leftArm && rightArm) {
      const leftBase = leftArm.quaternion.clone();
      const rightBase = rightArm.quaternion.clone();
      const swingFwd = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.6, 0, 0));
      const swingBack = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.5, 0, 0));
      
      runTracks.push(new THREE.QuaternionKeyframeTrack(
        `${leftArm.name}.quaternion`,
        [0, 0.3, 0.6],
        [
          ...leftBase.clone().multiply(swingFwd).toArray(),
          ...leftBase.clone().multiply(swingBack).toArray(),
          ...leftBase.clone().multiply(swingFwd).toArray()
        ]
      ));
      
      runTracks.push(new THREE.QuaternionKeyframeTrack(
        `${rightArm.name}.quaternion`,
        [0, 0.3, 0.6],
        [
          ...rightBase.clone().multiply(swingBack).toArray(),
          ...rightBase.clone().multiply(swingFwd).toArray(),
          ...rightBase.clone().multiply(swingBack).toArray()
        ]
      ));
    }
    
    if (leftLeg && rightLeg) {
      const leftBase = leftLeg.quaternion.clone();
      const rightBase = rightLeg.quaternion.clone();
      const legFwd = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.7, 0, 0));
      const legBack = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.5, 0, 0));
      
      runTracks.push(new THREE.QuaternionKeyframeTrack(
        `${leftLeg.name}.quaternion`,
        [0, 0.3, 0.6],
        [
          ...leftBase.clone().multiply(legFwd).toArray(),
          ...leftBase.clone().multiply(legBack).toArray(),
          ...leftBase.clone().multiply(legFwd).toArray()
        ]
      ));
      
      runTracks.push(new THREE.QuaternionKeyframeTrack(
        `${rightLeg.name}.quaternion`,
        [0, 0.3, 0.6],
        [
          ...rightBase.clone().multiply(legBack).toArray(),
          ...rightBase.clone().multiply(legFwd).toArray(),
          ...rightBase.clone().multiply(legBack).toArray()
        ]
      ));
    }
    
    if (runTracks.length > 0) {
      const runClip = new THREE.AnimationClip('ProceduralRun', runDuration, runTracks);
      this.actions.run = this.mixer.clipAction(runClip);
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // WAVE ANIMATION - Friendly greeting
    // ═══════════════════════════════════════════════════════════════════════
    if (rightArm) {
      const baseRot = rightArm.quaternion.clone();
      const raiseArm = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -2.5)); // Raise arm up
      const waveLeft = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.3, 0, -2.5));
      const waveRight = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.3, 0, -2.5));
      
      const waveTracks = [
        new THREE.QuaternionKeyframeTrack(
          `${rightArm.name}.quaternion`,
          [0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6],
          [
            ...baseRot.toArray(),
            ...baseRot.clone().multiply(raiseArm).toArray(),
            ...baseRot.clone().multiply(waveLeft).toArray(),
            ...baseRot.clone().multiply(waveRight).toArray(),
            ...baseRot.clone().multiply(waveLeft).toArray(),
            ...baseRot.clone().multiply(waveRight).toArray(),
            ...baseRot.clone().multiply(raiseArm).toArray(),
            ...baseRot.toArray()
          ]
        )
      ];
      
      const waveClip = new THREE.AnimationClip('ProceduralWave', 1.6, waveTracks);
      this.actions.wave = this.mixer.clipAction(waveClip);
      this.actions.wave.setLoop(THREE.LoopOnce);
      this.actions.wave.clampWhenFinished = true;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // DANCE ANIMATION - Fun celebratory motion
    // ═══════════════════════════════════════════════════════════════════════
    const danceTracks = [];
    const danceDuration = 2.0;
    
    if (rootBone) {
      const baseY = rootBone.position.y;
      danceTracks.push(new THREE.VectorKeyframeTrack(
        `${rootBone.name}.position`,
        [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        [
          rootBone.position.x, baseY, rootBone.position.z,
          rootBone.position.x, baseY + 0.05, rootBone.position.z,
          rootBone.position.x, baseY, rootBone.position.z,
          rootBone.position.x, baseY + 0.05, rootBone.position.z,
          rootBone.position.x, baseY, rootBone.position.z,
          rootBone.position.x, baseY + 0.05, rootBone.position.z,
          rootBone.position.x, baseY, rootBone.position.z,
          rootBone.position.x, baseY + 0.05, rootBone.position.z,
          rootBone.position.x, baseY, rootBone.position.z
        ]
      ));
    }
    
    if (leftArm && rightArm) {
      const leftBase = leftArm.quaternion.clone();
      const rightBase = rightArm.quaternion.clone();
      const armsUp = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -1.5));
      const armsDown = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0.3));
      
      danceTracks.push(new THREE.QuaternionKeyframeTrack(
        `${leftArm.name}.quaternion`,
        [0, 0.5, 1.0, 1.5, 2.0],
        [
          ...leftBase.toArray(),
          ...leftBase.clone().multiply(armsUp).toArray(),
          ...leftBase.clone().multiply(armsDown).toArray(),
          ...leftBase.clone().multiply(armsUp).toArray(),
          ...leftBase.toArray()
        ]
      ));
      
      danceTracks.push(new THREE.QuaternionKeyframeTrack(
        `${rightArm.name}.quaternion`,
        [0, 0.5, 1.0, 1.5, 2.0],
        [
          ...rightBase.toArray(),
          ...rightBase.clone().multiply(armsUp).toArray(),
          ...rightBase.clone().multiply(armsDown).toArray(),
          ...rightBase.clone().multiply(armsUp).toArray(),
          ...rightBase.toArray()
        ]
      ));
    }
    
    if (danceTracks.length > 0) {
      const danceClip = new THREE.AnimationClip('ProceduralDance', danceDuration, danceTracks);
      this.actions.dance = this.mixer.clipAction(danceClip);
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // JUMP ANIMATION
    // ═══════════════════════════════════════════════════════════════════════
    if (rootBone) {
      const baseY = rootBone.position.y;
      const jumpTracks = [
        new THREE.VectorKeyframeTrack(
          `${rootBone.name}.position`,
          [0, 0.15, 0.4, 0.6, 0.8],
          [
            rootBone.position.x, baseY, rootBone.position.z,        // Start
            rootBone.position.x, baseY - 0.02, rootBone.position.z, // Crouch
            rootBone.position.x, baseY + 0.15, rootBone.position.z, // Peak
            rootBone.position.x, baseY + 0.05, rootBone.position.z, // Falling
            rootBone.position.x, baseY, rootBone.position.z         // Land
          ]
        )
      ];
      
      const jumpClip = new THREE.AnimationClip('ProceduralJump', 0.8, jumpTracks);
      this.actions.jump = this.mixer.clipAction(jumpClip);
      this.actions.jump.setLoop(THREE.LoopOnce);
      this.actions.jump.clampWhenFinished = true;
    }
    
    console.log("Created procedural animations:", Object.keys(this.actions));
  }

  _setupControls() {
    // Keyboard controls
    window.addEventListener("keydown", (e) => {
      // Don't process if typing in input
      if (document.activeElement?.tagName === "INPUT" || 
          document.activeElement?.tagName === "TEXTAREA") return;
      
      // User input overrides AI control
      const isMovementKey = ["KeyW", "KeyS", "KeyA", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.code);
      if (isMovementKey && this.aiControlled) {
        console.log("👤 User taking control - stopping AI walk");
        this.stopWalking();
      }
      
      switch(e.code) {
        case "KeyW":
        case "ArrowUp":
          this.input.forward = true;
          break;
        case "KeyS":
        case "ArrowDown":
          this.input.backward = true;
          break;
        case "KeyA":
        case "ArrowLeft":
          this.input.left = true;
          break;
        case "KeyD":
        case "ArrowRight":
          this.input.right = true;
          break;
        case "ShiftLeft":
        case "ShiftRight":
          this.input.run = true;
          break;
        case "Space":
          if (!this.input.jump) {
            this.input.jump = true;
            this.playGesture("jump");
          }
          break;
        case "KeyG":
          this.playGesture("wave");
          break;
        case "KeyT":
          this.playGesture("talk");
          break;
        case "KeyY":
          this.playGesture("dance");
          break;
      }
    });
    
    window.addEventListener("keyup", (e) => {
      switch(e.code) {
        case "KeyW":
        case "ArrowUp":
          this.input.forward = false;
          break;
        case "KeyS":
        case "ArrowDown":
          this.input.backward = false;
          break;
        case "KeyA":
        case "ArrowLeft":
          this.input.left = false;
          break;
        case "KeyD":
        case "ArrowRight":
          this.input.right = false;
          break;
        case "ShiftLeft":
        case "ShiftRight":
          this.input.run = false;
          break;
        case "Space":
          this.input.jump = false;
          break;
      }
    });
  }

  _play(name) {
    if (!this.actions[name] || this.activeAction === this.actions[name]) return;
    if (this.activeAction) this.activeAction.fadeOut(0.25);
    this.activeAction = this.actions[name];
    this.currentAnimation = name;
    this.activeAction.reset().fadeIn(0.25).play();
  }
  
  playGesture(name) {
    if (!this.actions[name]) return;
    
    // Play gesture animation once, then return to previous state
    const previousAction = this.activeAction;
    const previousAnimation = this.currentAnimation;
    
    if (previousAction) previousAction.fadeOut(0.25);
    
    const gestureAction = this.actions[name];
    gestureAction.reset().fadeIn(0.25).play();
    
    // Listen for gesture completion
    const onFinish = () => {
      this.mixer.removeEventListener("finished", onFinish);
      if (previousAnimation && this.actions[previousAnimation]) {
        gestureAction.fadeOut(0.25);
        this.actions[previousAnimation].reset().fadeIn(0.25).play();
        this.activeAction = this.actions[previousAnimation];
        this.currentAnimation = previousAnimation;
      }
    };
    
    this.mixer.addEventListener("finished", onFinish);
  }

  teleport(position) {
    if (!this.model) return;
    this.model.position.set(position.x ?? 0, position.y ?? 0, position.z ?? 0);
    // Cancel any active walking
    this.walkTarget = null;
    this.aiControlled = false;
  }

  /**
   * OLD: Instant interpolated move (kept for backwards compatibility)
   */
  moveTo(position, duration = 0.75) {
    if (!this.model) return;
    const start = this.model.position.clone();
    const end = new THREE.Vector3(position.x ?? start.x, position.y ?? start.y, position.z ?? start.z);
    const delta = end.clone().sub(start);
    this.velocity.copy(delta).divideScalar(Math.max(duration, 0.01));
    this.target = { end, remaining: duration };
    this._play(delta.length() > 0.1 ? "walk" : "idle");
    
    // Face movement direction
    if (delta.length() > 0.01) {
      this.targetRotation = Math.atan2(delta.x, delta.z);
    }
  }

  /**
   * NEW: Realistic walking - AI takes control of inputs and walks step by step
   * This simulates pressing WASD keys to reach the destination naturally
   */
  walkTo(position, options = {}) {
    if (!this.model) return;
    
    const target = new THREE.Vector3(
      position.x ?? this.model.position.x,
      position.y ?? this.model.position.y,
      position.z ?? this.model.position.z
    );
    
    const distance = this.model.position.distanceTo(target);
    
    // Don't walk to very close targets
    if (distance < 0.3) {
      console.log("🚶 Already at destination");
      return;
    }
    
    this.walkTarget = {
      position: target,
      arrivalThreshold: options.arrivalThreshold ?? 0.5,  // How close to get
      shouldRun: options.run ?? (distance > 8),           // Run for long distances
      onArrival: options.onArrival ?? null,               // Callback when arrived
      startTime: Date.now(),
      startPosition: this.model.position.clone()
    };
    
    // AI takes control
    this.aiControlled = true;
    
    console.log(`🚶 Walking to (${target.x.toFixed(1)}, ${target.z.toFixed(1)}) - distance: ${distance.toFixed(1)}m`);
  }

  /**
   * Stop AI-controlled walking and return to idle
   */
  stopWalking() {
    this.walkTarget = null;
    this.aiControlled = false;
    this.input.forward = false;
    this.input.backward = false;
    this.input.left = false;
    this.input.right = false;
    this.input.run = false;
    this._play("idle");
    console.log("🛑 Stopped walking");
  }

  /**
   * Check if AI is currently walking to a target
   */
  isWalking() {
    return this.aiControlled && this.walkTarget !== null;
  }

  /**
   * Get current walk progress (0-1)
   */
  getWalkProgress() {
    if (!this.walkTarget || !this.model) return 1;
    const totalDist = this.walkTarget.startPosition.distanceTo(this.walkTarget.position);
    const remainingDist = this.model.position.distanceTo(this.walkTarget.position);
    return totalDist > 0 ? 1 - (remainingDist / totalDist) : 1;
  }

  push(vector) {
    this.velocity.add(new THREE.Vector3(vector.x ?? 0, vector.y ?? 0, vector.z ?? 0));
  }

  update(dt) {
    if (!this.model) return;

    // Handle scripted movement (old moveTo - instant interpolation)
    if (this.target) {
      const step = Math.min(this.target.remaining, dt);
      const movement = this.velocity.clone().multiplyScalar(step);
      this.model.position.add(movement);
      this.target.remaining -= step;
      if (this.target.remaining <= 0) {
        this.model.position.copy(this.target.end);
        this.velocity.set(0, 0, 0);
        this.target = null;
        this._play("idle");
      }
    } 
    // Handle AI-controlled realistic walking (walkTo)
    else if (this.aiControlled && this.walkTarget) {
      const target = this.walkTarget.position;
      const currentPos = this.model.position;
      
      // Calculate direction to target (2D, ignore Y)
      const dx = target.x - currentPos.x;
      const dz = target.z - currentPos.z;
      const distance = Math.sqrt(dx * dx + dz * dz);
      
      // Check if arrived
      if (distance < this.walkTarget.arrivalThreshold) {
        console.log("✅ Arrived at destination!");
        const callback = this.walkTarget.onArrival;
        this.stopWalking();
        if (callback) callback();
        return;
      }
      
      // Calculate desired facing angle
      const desiredAngle = Math.atan2(dx, dz);
      
      // Current facing angle
      let currentAngle = this.model.rotation.y;
      
      // Calculate angle difference
      let angleDiff = desiredAngle - currentAngle;
      while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
      while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
      
      // Simulate AI input based on direction
      // Reset inputs first
      this.input.forward = false;
      this.input.backward = false;
      this.input.left = false;
      this.input.right = false;
      
      // If we need to turn significantly, turn first before walking
      if (Math.abs(angleDiff) > 0.3) {
        // Turn toward target
        if (angleDiff > 0) {
          this.input.right = true;
        } else {
          this.input.left = true;
        }
        // Also move forward slightly while turning
        if (Math.abs(angleDiff) < 1.5) {
          this.input.forward = true;
        }
      } else {
        // Facing roughly correct direction, walk forward
        this.input.forward = true;
      }
      
      // Run for long distances
      this.input.run = this.walkTarget.shouldRun && distance > 3;
      
      // Now let the normal input handler process these AI inputs
      // Fall through to input handling below...
    }
    
    // Handle input movement (keyboard OR AI-controlled)
    if (!this.target) {
      const moveDir = new THREE.Vector3();
      
      if (this.input.forward) moveDir.z += 1;
      if (this.input.backward) moveDir.z -= 1;
      if (this.input.left) moveDir.x -= 1;
      if (this.input.right) moveDir.x += 1;
      
      this.isMoving = moveDir.length() > 0;
      this.isRunning = this.input.run && this.isMoving;
      
      if (this.isMoving) {
        moveDir.normalize();
        
        // Calculate target rotation from movement direction
        this.targetRotation = Math.atan2(moveDir.x, moveDir.z);
        
        // Smoothly rotate towards movement direction
        const currentRotation = this.model.rotation.y;
        let rotationDiff = this.targetRotation - currentRotation;
        
        // Normalize rotation difference to -PI to PI
        while (rotationDiff > Math.PI) rotationDiff -= Math.PI * 2;
        while (rotationDiff < -Math.PI) rotationDiff += Math.PI * 2;
        
        this.model.rotation.y += rotationDiff * this.options.turnSpeed * dt;
        
        // Move in the direction we're facing (model faces +Z)
        const speed = this.isRunning ? this.options.runSpeed : this.options.moveSpeed;
        const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(this.model.quaternion);
        this.model.position.addScaledVector(forward, speed * dt);
        
        // Play appropriate animation
        this._play(this.isRunning ? "run" : "walk");
      } else {
        // Not moving - play idle
        if (this.currentAnimation === "walk" || this.currentAnimation === "run") {
          this._play("idle");
        }
      }
    }
    
    // Smooth rotation towards target (when not moving)
    if (this.targetRotation !== undefined && !this.isMoving && !this.aiControlled) {
      const currentRotation = this.model.rotation.y;
      let rotationDiff = this.targetRotation - currentRotation;
      while (rotationDiff > Math.PI) rotationDiff -= Math.PI * 2;
      while (rotationDiff < -Math.PI) rotationDiff += Math.PI * 2;
      if (Math.abs(rotationDiff) > 0.01) {
        this.model.rotation.y += rotationDiff * this.options.turnSpeed * dt;
      }
    }

    // Update motor control if enabled (humanoid-style joint control)
    if (this.motorControl && this.useMotorControl) {
      this.motorControl.update(dt);
    }
    
    // Only update animations if not using motor control
    if (this.mixer && !this.useMotorControl) {
      this.mixer.update(dt);
    }
  }
  
  // ═══════════════════════════════════════════════════════════════════════════
  // MOTOR CONTROL API - Humanoid-style direct joint control
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Enable motor control mode (disable animations)
   */
  enableMotorControl() {
    if (!this.motorControl) {
      console.warn("Motor control not available");
      return;
    }
    this.useMotorControl = true;
    this.motorControl.enable();
    console.log("🦾 Motor control ENABLED - Aether now has direct joint control");
  }
  
  /**
   * Disable motor control mode (return to animations)
   */
  disableMotorControl() {
    this.useMotorControl = false;
    if (this.motorControl) {
      this.motorControl.disable();
    }
    console.log("🎬 Animation mode restored");
  }
  
  /**
   * Raise arm (0 = down, 1 = fully raised)
   */
  raiseArm(side, amount) {
    if (!this.motorControl || !this.useMotorControl) return;
    if (side === "left") {
      this.motorControl.raiseLeftArm(amount);
    } else {
      this.motorControl.raiseRightArm(amount);
    }
  }
  
  /**
   * Wave hand
   */
  waveHand(hand = "right") {
    if (!this.motorControl) return;
    this.enableMotorControl();
    
    // Animated wave
    let waveTime = 0;
    const waveInterval = setInterval(() => {
      waveTime += 0.1;
      this.motorControl.wave(hand, waveTime * 10);
      
      if (waveTime > 2) {
        clearInterval(waveInterval);
        this.disableMotorControl();
      }
    }, 50);
  }
  
  /**
   * Point at world position
   */
  pointAtPosition(position, hand = "right") {
    if (!this.motorControl) return;
    this.enableMotorControl();
    this.motorControl.pointAt(position, hand);
    this.motorControl.setHandPose(hand, "point");
  }
  
  /**
   * Look at world position (head tracking)
   */
  lookAtPosition(position) {
    if (!this.motorControl) return;
    this.enableMotorControl();
    this.motorControl.lookAt(position);
  }
  
  /**
   * Reach hand to position (IK)
   */
  reachTo(position, hand = "right") {
    if (!this.motorControl) return;
    this.enableMotorControl();
    this.motorControl.reachTo(position, hand);
    this.motorControl.setHandPose(hand, "grab");
  }
  
  /**
   * Set finger pose
   */
  setFingers(hand, pose) {
    if (!this.motorControl) return;
    this.enableMotorControl();
    this.motorControl.setHandPose(hand, pose);
  }
  
  /**
   * Get current body proprioception
   */
  getBodyState() {
    if (!this.motorControl) return null;
    return {
      jointAngles: this.motorControl.getJointAngles(),
      endEffectors: this.motorControl.getEndEffectorPositions(),
      pose: this.motorControl.describePose()
    };
  }

  serialize() {
    if (!this.model) return { active: false };
    return {
      active: true,
      position: {
        x: this.model.position.x,
        y: this.model.position.y,
        z: this.model.position.z
      },
      rotation: this.model.rotation.y,
      velocity: {
        x: this.velocity.x,
        y: this.velocity.y,
        z: this.velocity.z
      },
      animation: this.currentAnimation,
      isMoving: this.isMoving,
      isRunning: this.isRunning
    };
  }
}
