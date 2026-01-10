/**
 * VisionSystem.js
 * Captures camera frames and streams them to the backend for visual perception
 * Enables Aether to "see" and understand the 3D environment
 */

export class VisionSystem {
  constructor(renderer, camera, options = {}) {
    this.renderer = renderer;
    this.camera = camera;
    
    this.options = {
      captureWidth: 320,
      captureHeight: 240,
      captureInterval: 1000, // ms between captures
      quality: 0.7,
      apiUrl: import.meta.env.VITE_AETHER_API_URL?.replace('/v1/game/unity/state', '/v1/perception/vision') 
              ?? "http://localhost:8000/v1/perception/vision",
      apiKey: import.meta.env.VITE_AETHER_API_KEY ?? "dev_mode",
      enabled: true,
      ...options
    };
    
    // Off-screen canvas for capture
    this.captureCanvas = document.createElement("canvas");
    this.captureCanvas.width = this.options.captureWidth;
    this.captureCanvas.height = this.options.captureHeight;
    this.captureCtx = this.captureCanvas.getContext("2d");
    
    // Capture timer
    this.captureTimer = null;
    this.lastCapture = null;
    this.lastVisionResult = null;
    
    // Stream state
    this.isStreaming = false;
    this.frameCount = 0;
    
    // Camera control state
    this.cameraYawOffset = 0;    // Left/right rotation
    this.cameraPitchOffset = 0;  // Up/down rotation
    this.originalCameraRotation = null;
    this.isLooking = false;
    
    // Multi-angle vision cache
    this.multiAngleCache = {
      front: null,
      left: null,
      right: null,
      up: null,
      down: null
    };
    
    console.log("👁️ Vision system initialized with active camera control");
  }
  
  /**
   * Start streaming vision to backend
   */
  start() {
    if (this.isStreaming) return;
    
    this.isStreaming = true;
    this.captureTimer = setInterval(() => {
      this.captureAndSend();
    }, this.options.captureInterval);
    
    console.log("👁️ Vision streaming started");
  }
  
  /**
   * Stop streaming
   */
  stop() {
    if (!this.isStreaming) return;
    
    clearInterval(this.captureTimer);
    this.captureTimer = null;
    this.isStreaming = false;
    
    console.log("👁️ Vision streaming stopped");
  }
  
  /**
   * Capture current frame
   */
  capture() {
    // Render to main canvas first
    const mainCanvas = this.renderer.domElement;
    
    // Draw scaled down version
    this.captureCtx.drawImage(
      mainCanvas,
      0, 0, mainCanvas.width, mainCanvas.height,
      0, 0, this.options.captureWidth, this.options.captureHeight
    );
    
    // Get as base64 JPEG
    const dataUrl = this.captureCanvas.toDataURL("image/jpeg", this.options.quality);
    const base64 = dataUrl.split(",")[1];
    
    this.lastCapture = {
      data: base64,
      width: this.options.captureWidth,
      height: this.options.captureHeight,
      timestamp: Date.now(),
      frameNumber: this.frameCount++
    };
    
    return this.lastCapture;
  }
  
  /**
   * Capture and send to backend
   */
  async captureAndSend() {
    if (!this.options.enabled) return;
    
    const frame = this.capture();
    
    try {
      const response = await fetch(this.options.apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Aether-Key": this.options.apiKey
        },
        body: JSON.stringify({
          frame: frame.data,
          width: frame.width,
          height: frame.height,
          timestamp: frame.timestamp,
          frameNumber: frame.frameNumber
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        this.lastVisionResult = result;
        return result;
      }
    } catch (error) {
      // Silently fail - vision is optional
      console.debug("Vision stream error:", error.message);
    }
    
    return null;
  }
  
  /**
   * Get single frame analysis (on-demand)
   */
  async analyzeFrame(prompt = "What do you see?") {
    const frame = this.capture();
    
    try {
      const response = await fetch(this.options.apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Aether-Key": this.options.apiKey
        },
        body: JSON.stringify({
          frame: frame.data,
          width: frame.width,
          height: frame.height,
          timestamp: frame.timestamp,
          prompt: prompt,
          detailed: true
        })
      });
      
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error("Vision analysis error:", error);
    }
    
    return null;
  }
  
  /**
   * Get last vision result
   */
  getLastResult() {
    return this.lastVisionResult;
  }
  
  /**
   * Get current view as description for AI
   */
  describeView() {
    if (!this.lastVisionResult) {
      return "No visual data available yet.";
    }
    
    const result = this.lastVisionResult;
    const parts = [];
    
    if (result.objects && result.objects.length > 0) {
      const objects = result.objects.slice(0, 5).map(o => o.label || o.name);
      parts.push(`I can see: ${objects.join(", ")}`);
    }
    
    if (result.scene_type) {
      parts.push(`The scene appears to be ${result.scene_type}`);
    }
    
    if (result.description) {
      parts.push(result.description);
    }
    
    return parts.length > 0 ? parts.join(". ") : "Visual processing in progress...";
  }
  
  // ========================================================================
  // ACTIVE CAMERA CONTROL - Look around the environment
  // ========================================================================
  
  /**
   * Rotate camera to look in a direction
   * @param {string} direction - "left", "right", "up", "down", "front"
   * @param {number} degrees - How much to rotate (default: 45)
   */
  lookInDirection(direction, degrees = 45) {
    if (!this.originalCameraRotation) {
      this.originalCameraRotation = {
        x: this.camera.rotation.x,
        y: this.camera.rotation.y,
        z: this.camera.rotation.z
      };
    }
    
    this.isLooking = true;
    const radians = (degrees * Math.PI) / 180;
    
    switch (direction.toLowerCase()) {
      case "left":
        this.cameraYawOffset = radians;
        break;
      case "right":
        this.cameraYawOffset = -radians;
        break;
      case "up":
        this.cameraPitchOffset = radians;
        break;
      case "down":
        this.cameraPitchOffset = -radians;
        break;
      case "front":
      case "center":
      case "forward":
        this.cameraYawOffset = 0;
        this.cameraPitchOffset = 0;
        break;
    }
    
    this.updateCameraRotation();
    console.log(`👁️ Looking ${direction} (${degrees}°)`);
  }
  
  /**
   * Apply camera rotation offsets
   */
  updateCameraRotation() {
    if (!this.originalCameraRotation) return;
    
    // Apply offsets to original rotation
    this.camera.rotation.y = this.originalCameraRotation.y + this.cameraYawOffset;
    this.camera.rotation.x = this.originalCameraRotation.x + this.cameraPitchOffset;
  }
  
  /**
   * Reset camera to original view
   */
  resetCamera() {
    if (this.originalCameraRotation) {
      this.camera.rotation.x = this.originalCameraRotation.x;
      this.camera.rotation.y = this.originalCameraRotation.y;
      this.camera.rotation.z = this.originalCameraRotation.z;
    }
    
    this.cameraYawOffset = 0;
    this.cameraPitchOffset = 0;
    this.isLooking = false;
    this.originalCameraRotation = null;
    
    console.log("👁️ Camera reset to original view");
  }
  
  /**
   * Scan area by capturing multiple angles
   * @param {Array} directions - Array of directions to scan ["left", "front", "right"]
   * @returns {Object} Multi-angle vision data
   */
  async scanArea(directions = ["left", "front", "right"]) {
    console.log(`👁️ Scanning area: ${directions.join(", ")}`);
    
    const results = {};
    
    for (const direction of directions) {
      // Look in direction
      this.lookInDirection(direction);
      
      // Wait a moment for camera to stabilize
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Capture and analyze
      const frame = this.capture();
      this.multiAngleCache[direction] = frame;
      
      // Optionally analyze each angle
      // const analysis = await this.analyzeFrame(`What do you see looking ${direction}?`);
      // results[direction] = analysis;
      
      results[direction] = {
        captured: true,
        timestamp: frame.timestamp
      };
    }
    
    // Reset camera
    this.resetCamera();
    
    return {
      success: true,
      directions: directions,
      results: results,
      timestamp: Date.now()
    };
  }
  
  /**
   * Focus on a specific point in 3D space
   * @param {Object} targetPosition - {x, y, z} world position
   */
  focusOnPoint(targetPosition) {
    if (!targetPosition) return;
    
    // Calculate direction to target
    const direction = {
      x: targetPosition.x - this.camera.position.x,
      y: targetPosition.y - this.camera.position.y,
      z: targetPosition.z - this.camera.position.z
    };
    
    // Calculate angles
    const yaw = Math.atan2(direction.x, direction.z);
    const distance = Math.sqrt(direction.x * direction.x + direction.z * direction.z);
    const pitch = Math.atan2(direction.y, distance);
    
    // Store original if not already looking
    if (!this.originalCameraRotation) {
      this.originalCameraRotation = {
        x: this.camera.rotation.x,
        y: this.camera.rotation.y,
        z: this.camera.rotation.z
      };
    }
    
    // Apply rotation
    this.camera.rotation.y = yaw;
    this.camera.rotation.x = -pitch;
    this.isLooking = true;
    
    console.log(`👁️ Focusing on point (${targetPosition.x.toFixed(1)}, ${targetPosition.y.toFixed(1)}, ${targetPosition.z.toFixed(1)})`);
  }
  
  /**
   * Capture panoramic view (360° scan)
   * @param {number} steps - Number of angles to capture (default: 8)
   */
  async capturePanorama(steps = 8) {
    console.log(`👁️ Capturing ${steps}-angle panorama`);
    
    const angleStep = 360 / steps;
    const captures = [];
    
    for (let i = 0; i < steps; i++) {
      const angle = i * angleStep;
      this.lookInDirection("front", 0);
      this.cameraYawOffset = (angle * Math.PI) / 180;
      this.updateCameraRotation();
      
      await new Promise(resolve => setTimeout(resolve, 50));
      
      const frame = this.capture();
      captures.push({
        angle: angle,
        frame: frame
      });
    }
    
    this.resetCamera();
    
    return {
      success: true,
      panorama: captures,
      steps: steps,
      timestamp: Date.now()
    };
  }
  
  /**
   * Get peripheral vision (what's to the sides)
   */
  async capturePeripheralVision() {
    return await this.scanArea(["left", "front", "right"]);
  }
  
  /**
   * Get vertical scan (look up and down)
   */
  async captureVerticalScan() {
    return await this.scanArea(["up", "front", "down"]);
  }
}
