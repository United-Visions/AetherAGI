import "./styles.css";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { SceneRegistry } from "./environment/SceneRegistry.js";
import { PlayerRig } from "./player/PlayerRig.js";
import { AetherBridge } from "./services/AetherBridge.js";
import { EmbodimentController } from "./body/EmbodimentController.js";

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
// Light sky blue background
scene.background = new THREE.Color(0xadd8e6);
// Light fog for depth
scene.fog = new THREE.Fog(0xadd8e6, 30, 100);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 500);

// Camera system state
let cameraMode = "third"; // "first" or "third"
// Offsets scaled for 0.004 scale Aether (roughly 0.7 units tall)
const THIRD_PERSON_OFFSET = new THREE.Vector3(0, 0.5, 1.5); // Behind and slightly above
const FIRST_PERSON_OFFSET = new THREE.Vector3(0, 0.6, 0); // At head height (Aether ~0.7 tall)

const controls = new OrbitControls(camera, renderer.domElement);
controls.maxPolarAngle = Math.PI / 2.1;
controls.minDistance = 0.5;
controls.maxDistance = 8;
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.enabled = true;

// Soft ambient for shadows
const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);

// Hemisphere light for natural sky/ground lighting
const hemi = new THREE.HemisphereLight(0x87ceeb, 0x3d5c47, 0.5);
scene.add(hemi);

// Strong sun with shadows
const sun = new THREE.DirectionalLight(0xfff5e1, 1.2);
sun.position.set(30, 50, 30);
sun.castShadow = true;
sun.shadow.mapSize.width = 2048;
sun.shadow.mapSize.height = 2048;
sun.shadow.camera.near = 0.5;
sun.shadow.camera.far = 80;
sun.shadow.camera.left = -25;
sun.shadow.camera.right = 25;
sun.shadow.camera.top = 25;
sun.shadow.camera.bottom = -25;
sun.shadow.bias = -0.0001;
scene.add(sun);

// Simple infinite ground plane
const groundGeo = new THREE.PlaneGeometry(200, 200);
const groundMat = new THREE.MeshStandardMaterial({ 
  color: 0x88bb88, // Grass green
  roughness: 0.9
});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.y = 0;
ground.receiveShadow = true;
scene.add(ground);

// Add a grid helper for reference
const gridHelper = new THREE.GridHelper(50, 50, 0x444444, 0x888888);
gridHelper.position.y = 0.01;
scene.add(gridHelper);

const registry = new SceneRegistry(scene);

// Just Aether - no city, no demo objects
const player = new PlayerRig(scene, registry, {
  characterAsset: "/assets/characters/aether.glb",
  animationLibrary: "/assets/animations/universal_animations.glb",
  scale: 0.004, // Human size
  initialPosition: { x: 0, y: 0, z: 0 }
});
registry.track("player", player);

const bridge = new AetherBridge({
  registry,
  player,
  fetchInterval: Number(import.meta.env.VITE_AETHER_SYNC_INTERVAL ?? 500)
});

// Initialize embodiment (self-awareness) system
const embodiment = new EmbodimentController({
  enableAutonomous: false, // Start manual
  enableExploration: false // No exploration in empty space
});

// Initialize embodiment after player loads
setTimeout(async () => {
  if (player.model) {
    await embodiment.initialize(scene, renderer, camera, player);
    console.log("🧠 Embodiment ready - Aether can now feel its body!");
    
    // Callback when Aether says something
    embodiment.onAction = (action) => {
      if (action.type === "speech") {
        showResponse(action.message);
      }
    };
    
    // Callback when Aether has a thought
    embodiment.onThought = (thought) => {
      console.log("💭 Aether thought:", thought);
    };
  }
}, 1000);

// ═══════════════════════════════════════════════════════════════════════════
// COMPACT CHAT UI WITH SELF-AWARE MODE TOGGLE
// ═══════════════════════════════════════════════════════════════════════════

// Self-aware mode state
let isAutonomous = false;

const uiContainer = document.createElement("div");
uiContainer.style.cssText = `
  position: fixed;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  z-index: 1000;
  font-family: 'Segoe UI', system-ui, sans-serif;
`;

// Response display (shows above chat)
const responseDisplay = document.createElement("div");
responseDisplay.style.cssText = `
  width: 480px;
  max-width: 92vw;
  max-height: 120px;
  overflow-y: auto;
  padding: 10px 14px;
  background: rgba(0, 0, 0, 0.75);
  border-radius: 12px;
  backdrop-filter: blur(12px);
  color: white;
  font-size: 13px;
  line-height: 1.4;
  display: none;
  border: 1px solid rgba(75, 193, 238, 0.2);
`;

// Main chat bar container
const chatBar = document.createElement("div");
chatBar.style.cssText = `
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px;
  background: rgba(0, 0, 0, 0.8);
  border-radius: 28px;
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

// Self-aware toggle button (robot icon)
const selfAwareBtn = document.createElement("button");
selfAwareBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
  <rect x="3" y="4" width="18" height="12" rx="2"/>
  <circle cx="9" cy="10" r="1.5" fill="currentColor"/>
  <circle cx="15" cy="10" r="1.5" fill="currentColor"/>
  <path d="M8 16v2m8-2v2"/>
  <path d="M12 2v2"/>
  <path d="M7 20h2m6 0h2"/>
</svg>`;
selfAwareBtn.title = "Toggle Self-Aware Mode";
selfAwareBtn.style.cssText = `
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  background: rgba(255, 255, 255, 0.1);
  color: #888;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  flex-shrink: 0;
`;

// Glow animation for active state
const glowKeyframes = `
  @keyframes robotGlow {
    0%, 100% { box-shadow: 0 0 8px #4bc1ee, 0 0 16px rgba(75, 193, 238, 0.4); }
    50% { box-shadow: 0 0 16px #4bc1ee, 0 0 32px rgba(75, 193, 238, 0.6); }
  }
`;
const styleSheet = document.createElement("style");
styleSheet.textContent = glowKeyframes;
document.head.appendChild(styleSheet);

function updateSelfAwareButton() {
  if (isAutonomous) {
    selfAwareBtn.style.background = "rgba(75, 193, 238, 0.3)";
    selfAwareBtn.style.color = "#4bc1ee";
    selfAwareBtn.style.animation = "robotGlow 2s ease-in-out infinite";
    selfAwareBtn.title = "Self-Aware Mode ON (click to disable)";
  } else {
    selfAwareBtn.style.background = "rgba(255, 255, 255, 0.1)";
    selfAwareBtn.style.color = "#888";
    selfAwareBtn.style.animation = "none";
    selfAwareBtn.style.boxShadow = "none";
    selfAwareBtn.title = "Enable Self-Aware Mode";
  }
}

selfAwareBtn.addEventListener("click", () => {
  if (!embodiment.initialized) {
    showResponse("⏳ Embodiment system still initializing...");
    return;
  }
  
  isAutonomous = !isAutonomous;
  
  if (isAutonomous) {
    embodiment.options.enableAutonomous = true;
    embodiment.startAutonomousLoop();
    showResponse("🤖 <strong>Self-Aware Mode ON</strong> — Aether can now think and act autonomously!");
  } else {
    embodiment.stopAutonomousLoop();
    showResponse("🎮 <strong>Manual Mode</strong> — You have full control.");
  }
  
  updateSelfAwareButton();
});

selfAwareBtn.addEventListener("mouseenter", () => {
  if (!isAutonomous) {
    selfAwareBtn.style.background = "rgba(75, 193, 238, 0.2)";
    selfAwareBtn.style.color = "#4bc1ee";
  }
});

selfAwareBtn.addEventListener("mouseleave", () => {
  if (!isAutonomous) {
    selfAwareBtn.style.background = "rgba(255, 255, 255, 0.1)";
    selfAwareBtn.style.color = "#888";
  }
});

// Chat input
const chatInput = document.createElement("input");
chatInput.type = "text";
chatInput.placeholder = "Talk to Aether... (try: raise your right arm, wave, point at me)";
chatInput.style.cssText = `
  flex: 1;
  min-width: 280px;
  max-width: 360px;
  padding: 10px 16px;
  border: none;
  border-radius: 20px;
  font-size: 14px;
  background: rgba(255, 255, 255, 0.08);
  color: white;
  outline: none;
  transition: all 0.2s ease;
`;
chatInput.addEventListener("focus", () => {
  chatInput.style.background = "rgba(255, 255, 255, 0.12)";
  chatInput.style.boxShadow = "0 0 0 2px rgba(75, 193, 238, 0.4)";
});
chatInput.addEventListener("blur", () => {
  chatInput.style.background = "rgba(255, 255, 255, 0.08)";
  chatInput.style.boxShadow = "none";
});

// Send button
const sendBtn = document.createElement("button");
sendBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
</svg>`;
sendBtn.style.cssText = `
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  background: #4bc1ee;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  flex-shrink: 0;
`;
sendBtn.addEventListener("mouseenter", () => {
  sendBtn.style.background = "#3aa8d4";
  sendBtn.style.transform = "scale(1.05)";
});
sendBtn.addEventListener("mouseleave", () => {
  sendBtn.style.background = "#4bc1ee";
  sendBtn.style.transform = "scale(1)";
});

// Camera mode toggle (compact)
const cameraToggle = document.createElement("button");
cameraToggle.innerHTML = "👁️";
cameraToggle.title = "Toggle Camera View (V)";
cameraToggle.style.cssText = `
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  background: rgba(255, 255, 255, 0.1);
  font-size: 16px;
  transition: all 0.2s ease;
  flex-shrink: 0;
`;

function updateCameraToggle() {
  cameraToggle.innerHTML = cameraMode === "first" ? "👁️" : "🎮";
  cameraToggle.title = cameraMode === "first" ? "First Person (press V)" : "Third Person (press V)";
}

cameraToggle.addEventListener("click", () => {
  setCameraMode(cameraMode === "first" ? "third" : "first");
  updateCameraToggle();
});

cameraToggle.addEventListener("mouseenter", () => {
  cameraToggle.style.background = "rgba(255, 255, 255, 0.2)";
});
cameraToggle.addEventListener("mouseleave", () => {
  cameraToggle.style.background = "rgba(255, 255, 255, 0.1)";
});

// Assemble chat bar
chatBar.appendChild(selfAwareBtn);
chatBar.appendChild(chatInput);
chatBar.appendChild(sendBtn);
chatBar.appendChild(cameraToggle);

uiContainer.appendChild(responseDisplay);
uiContainer.appendChild(chatBar);
document.body.appendChild(uiContainer);

// Helper function to show responses in the UI
function showResponse(message) {
  responseDisplay.style.display = "block";
  responseDisplay.innerHTML = message;
  
  // Auto-hide after 6 seconds for short messages
  if (message.length < 150 && !message.includes("Self-Aware")) {
    setTimeout(() => {
      if (responseDisplay.innerHTML === message) {
        responseDisplay.style.display = "none";
      }
    }, 6000);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// AETHER CHAT - Connect to GAME CHAT endpoint for AI-controlled movement
// ═══════════════════════════════════════════════════════════════════════════

const CHAT_API_URL = import.meta.env.VITE_AETHER_CHAT_URL ?? "http://localhost:8000/v1/game/chat";
const QUEST_API_URL = import.meta.env.VITE_AETHER_API_URL?.replace('/v1/game/unity/state', '/v1/game') ?? "http://localhost:8000/v1/game";
const API_KEY = import.meta.env.VITE_AETHER_API_KEY ?? "dev_mode";

// Quest state
let activeQuest = null;
let explorationMode = false;

async function sendMessage(message) {
  if (!message.trim()) return;
  
  // Check for special commands
  const lowerMsg = message.toLowerCase().trim();
  
  // Goal command
  if (lowerMsg.startsWith("goal:") || lowerMsg.startsWith("goal ")) {
    const goal = message.slice(lowerMsg.indexOf(":") + 1 || 5).trim();
    if (goal && embodiment.initialized) {
      embodiment.setGoal(goal);
      showResponse(`🎯 <strong>Goal set:</strong> "${goal}"`);
      chatInput.value = "";
      
      // Auto-enable self-aware mode if not already
      if (!isAutonomous) {
        isAutonomous = true;
        embodiment.options.enableAutonomous = true;
        embodiment.startAutonomousLoop();
        updateSelfAwareButton();
        showResponse(`🎯 <strong>Goal set:</strong> "${goal}"<br><small style="color:#4bc1ee;">Self-Aware Mode enabled automatically</small>`);
      }
      return;
    }
  }
  
  // Show thinking state
  responseDisplay.style.display = "block";
  responseDisplay.innerHTML = `<span style="color:#888;">💭 Aether is thinking...</span>`;
  chatInput.disabled = true;
  sendBtn.disabled = true;
  
  try {
    // Build context including motor control state
    const context = {
      interface: "3d_environment",
      self_aware_mode: isAutonomous,
      motor_control_enabled: player.useMotorControl || false,
      player_position: player.model ? {
        x: player.model.position.x,
        y: player.model.position.y,
        z: player.model.position.z
      } : null,
      animation: player.currentAnimation || "idle",
      is_walking: player.isWalking ? player.isWalking() : false
    };
    
    // Add body state if motor control is active
    if (player.useMotorControl && player.motorControl) {
      const bodyState = player.getBodyState();
      if (bodyState) {
        context.body_state = bodyState;
        context.pose_description = bodyState.pose;
      }
    }
    
    const response = await fetch(CHAT_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Aether-Key": API_KEY
      },
      body: JSON.stringify({
        message: message,
        context: context
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const data = await response.json();
    const reply = data.response || data.message || data.content || "...";
    
    // Display Aether's response
    responseDisplay.innerHTML = `<strong style="color:#4bc1ee;">Aether:</strong> ${reply}`;
    
    // Execute game commands from AI response
    if (data.commands && Array.isArray(data.commands)) {
      console.log("🎮 AI Commands:", data.commands);
      data.commands.forEach(cmd => bridge.execute(cmd));
    }
    
    // Update quest display
    if (data.active_quest) {
      activeQuest = data.active_quest;
      updateQuestDisplay();
    }
    
  } catch (error) {
    console.error("Chat error:", error);
    responseDisplay.innerHTML = `<span style="color: #ff6b6b;">Could not reach Aether. Is the backend running?</span>`;
  } finally {
    chatInput.disabled = false;
    sendBtn.disabled = false;
    chatInput.value = "";
    chatInput.focus();
  }
}

// Send on click or Enter
sendBtn.addEventListener("click", () => sendMessage(chatInput.value));
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(chatInput.value);
  }
});

function updateButtonStyles() {
  [firstPersonBtn, thirdPersonBtn].forEach(btn => {
    const isActive = btn.dataset.mode === cameraMode;
    btn.style.background = isActive ? "#4bc1ee" : "rgba(255,255,255,0.15)";
  });
}

function setCameraMode(mode) {
  cameraMode = mode;
  updateButtonStyles();
  
  if (mode === "first") {
    controls.enabled = false;
    // Hide Aether's model in first person
    if (player.model) player.model.visible = false;
  } else {
    controls.enabled = true;
    controls.minDistance = 0.8;
    controls.maxDistance = 5;
    // Show Aether's model in third person
    if (player.model) player.model.visible = true;
  }
}

// Keyboard shortcut: V to toggle view
window.addEventListener("keydown", (e) => {
  // Only trigger if not typing in chat
  if (document.activeElement === chatInput) return;
  if (e.key === "v" || e.key === "V") {
    setCameraMode(cameraMode === "first" ? "third" : "first");
  }
  
  // Motor control test keys (number keys)
  if (e.key === "1") {
    // Raise right arm
    player.enableMotorControl?.();
    player.raiseArm?.("right", 1);
    showResponse("🦾 Right arm raised (motor control)");
  }
  if (e.key === "2") {
    // Wave
    player.waveHand?.("right");
    showResponse("👋 Waving (motor control)");
  }
  if (e.key === "3") {
    // Point at camera
    player.pointAtPosition?.(camera.position, "right");
    showResponse("👆 Pointing at you (motor control)");
  }
  if (e.key === "4") {
    // Peace sign
    player.enableMotorControl?.();
    player.raiseArm?.("right", 0.8);
    player.setFingers?.("right", "peace");
    showResponse("✌️ Peace sign (motor control)");
  }
  if (e.key === "5") {
    // Thumbs up
    player.enableMotorControl?.();
    player.raiseArm?.("right", 0.5);
    player.setFingers?.("right", "thumbsUp");
    showResponse("👍 Thumbs up (motor control)");
  }
  if (e.key === "0") {
    // Return to animations
    player.disableMotorControl?.();
    showResponse("🎬 Animation mode restored");
  }
});

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener("resize", onWindowResize);

const clock = new THREE.Clock();

// Camera follow vectors
const cameraTarget = new THREE.Vector3();
const cameraPosition = new THREE.Vector3();
const smoothCameraPos = new THREE.Vector3();

function updateCamera() {
  if (!player.model) return;
  
  const playerPos = player.model.position;
  
  // NOTE: The Aether model faces +Z when rotation.y = 0
  // So "forward" is +Z, and "behind" for camera is -Z relative to model
  
  if (cameraMode === "first") {
    // First person: Camera at Aether's head level, looking forward
    const eyePos = playerPos.clone().add(new THREE.Vector3(0, FIRST_PERSON_OFFSET.y, 0));
    camera.position.lerp(eyePos, 0.2);
    
    // Look in the direction Aether is facing (+Z when rotation.y = 0)
    const lookDir = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), player.model.rotation.y);
    const lookAt = camera.position.clone().add(lookDir.multiplyScalar(10));
    camera.lookAt(lookAt);
  } else {
    // Third person: Camera BEHIND Aether based on her rotation
    // Behind = negative Z direction relative to model facing
    const behindOffset = new THREE.Vector3(
      -Math.sin(player.model.rotation.y) * THIRD_PERSON_OFFSET.z,
      THIRD_PERSON_OFFSET.y,
      -Math.cos(player.model.rotation.y) * THIRD_PERSON_OFFSET.z
    );
    
    cameraTarget.copy(playerPos).add(behindOffset);
    
    // Smooth camera follow
    smoothCameraPos.lerp(cameraTarget, 0.08);
    camera.position.copy(smoothCameraPos);
    
    // Always look at Aether (slightly above her center)
    const lookAtPoint = playerPos.clone().add(new THREE.Vector3(0, 0.35, 0));
    camera.lookAt(lookAtPoint);
    
    // Update orbit controls target to follow player
    controls.target.copy(lookAtPoint);
  }
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  
  updateCamera();
  
  if (cameraMode === "third") {
    controls.update();
  }
  
  player.update(delta);
  
  // Update embodiment systems (physics, body tracking)
  if (embodiment.initialized) {
    embodiment.update(delta);
  }
  
  renderer.render(scene, camera);
}

animate();
bridge.start();

// Initialize camera position behind Aether
setTimeout(() => {
  if (player.model) {
    const pos = player.model.position;
    const rot = player.model.rotation.y;
    // Position camera behind Aether (negative Z relative to facing direction)
    smoothCameraPos.set(
      pos.x - Math.sin(rot) * THIRD_PERSON_OFFSET.z,
      pos.y + THIRD_PERSON_OFFSET.y,
      pos.z - Math.cos(rot) * THIRD_PERSON_OFFSET.z
    );
    camera.position.copy(smoothCameraPos);
    controls.target.set(pos.x, pos.y + 0.35, pos.z);
    controls.update();
  }
}, 500);

// ═══════════════════════════════════════════════════════════════════════════
// COMPACT CONTROLS HELP (collapsible)
// ═══════════════════════════════════════════════════════════════════════════

const controlsHelp = document.createElement("div");
controlsHelp.style.cssText = `
  position: fixed;
  top: 16px;
  left: 16px;
  padding: 10px 14px;
  background: rgba(0, 0, 0, 0.75);
  border-radius: 12px;
  backdrop-filter: blur(10px);
  color: white;
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 11px;
  line-height: 1.6;
  z-index: 1000;
  max-width: 160px;
  border: 1px solid rgba(255, 255, 255, 0.1);
`;
controlsHelp.innerHTML = `
  <div style="font-weight: 600; margin-bottom: 6px; color: #4bc1ee; font-size: 12px;">🎮 Controls</div>
  <div><kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">WASD</kbd> Move</div>
  <div><kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">Shift</kbd> Run</div>
  <div><kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">Space</kbd> Jump</div>
  <div><kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">G</kbd> Wave · <kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">Y</kbd> Dance</div>
  <div><kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">V</kbd> Camera</div>
  <div style="margin-top: 6px; font-weight: 600; color: #ffcc00; font-size: 10px;">🦾 Motor Control</div>
  <div><kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">1</kbd> Raise Arm</div>
  <div><kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">2</kbd> Wave</div>
  <div><kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">3</kbd> Point</div>
  <div><kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">4</kbd> Peace ✌️</div>
  <div><kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">5</kbd> Thumbs Up</div>
  <div><kbd style="background:#333;padding:1px 4px;border-radius:2px;font-size:10px;">0</kbd> Reset</div>
`;
document.body.appendChild(controlsHelp);

// ═══════════════════════════════════════════════════════════════════════════
// INFO PANEL - Motor control status
// ═══════════════════════════════════════════════════════════════════════════

const infoPanel = document.createElement("div");
infoPanel.style.cssText = `
  position: fixed;
  top: 16px;
  right: 16px;
  padding: 10px 14px;
  background: rgba(0, 0, 0, 0.75);
  border-radius: 12px;
  backdrop-filter: blur(10px);
  color: white;
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 11px;
  z-index: 1000;
  border: 1px solid rgba(75, 193, 238, 0.2);
`;
infoPanel.innerHTML = `
  <div style="font-weight: 600; color: #4bc1ee; margin-bottom: 6px; font-size: 12px;">🦾 Motor Control</div>
  <div style="color: #888;">Press number keys to test</div>
  <div style="color: #888; margin-top: 4px;">Or talk to Aether in chat</div>
`;
document.body.appendChild(infoPanel);
