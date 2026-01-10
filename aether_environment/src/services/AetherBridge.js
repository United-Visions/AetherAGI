const API_URL = import.meta.env.VITE_AETHER_API_URL ?? "http://localhost:8000/v1/game/unity/state";
const API_KEY = import.meta.env.VITE_AETHER_API_KEY ?? "dev_mode";

export class AetherBridge {
  constructor({ registry, player, fetchInterval = 500 }) {
    this.registry = registry;
    this.player = player;
    this.fetchInterval = fetchInterval;
    this.timer = null;
  }

  start() {
    if (this.timer) return;
    this.timer = setInterval(() => this.sync(), this.fetchInterval);
  }

  stop() {
    if (!this.timer) return;
    clearInterval(this.timer);
    this.timer = null;
  }

  async sync() {
    try {
      const state = {
        timestamp: Date.now(),
        entities: this.registry.snapshot(),
        events: []
      };

      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Aether-Key": API_KEY
        },
        body: JSON.stringify(state)
      });

      if (!response.ok) {
        console.warn("Aether sync failed", response.status);
        return;
      }

      const data = await response.json();
      if (Array.isArray(data.commands)) {
        data.commands.forEach(cmd => this.execute(cmd));
      }
    } catch (error) {
      console.warn("Aether sync error", error);
    }
  }

  execute(command) {
    const { action, target, params = {} } = command || {};
    if (!action) return;

    switch (action) {
      case "move":
        // Use realistic walking (step by step) instead of instant move
        if (target === "player" && params.position) {
          // Use walkTo for realistic AI-controlled walking
          this.player.walkTo(params.position, {
            run: params.run ?? false,
            arrivalThreshold: params.arrivalThreshold ?? 0.5,
            onArrival: () => {
              console.log("🎯 AI walk complete");
            }
          });
        }
        break;
      case "move_instant":
        // Keep instant move available for special cases
        if (target === "player" && params.position) {
          this.player.moveTo(params.position, params.duration ?? 0.75);
        }
        break;
      case "teleport":
        if (target === "player" && params.position) {
          this.player.teleport(params.position);
        }
        break;
      case "stop":
        // Allow AI to stop walking
        if (target === "player") {
          this.player.stopWalking();
        }
        break;
      case "animate":
        if (target === "player" && params.animation) {
          // Stop walking if starting a gesture
          if (this.player.isWalking && this.player.isWalking()) {
            this.player.stopWalking();
          }
          // Play the specified animation
          if (typeof this.player.playGesture === "function") {
            this.player.playGesture(params.animation);
          } else if (typeof this.player._play === "function") {
            this.player._play(params.animation);
          }
          console.log(`🎬 Playing animation: ${params.animation}`);
        }
        break;
        
      // ═══════════════════════════════════════════════════════════════════════
      // MOTOR CONTROL COMMANDS - Humanoid-style direct joint control
      // ═══════════════════════════════════════════════════════════════════════
      
      case "raise_arm":
        // raise_arm { hand: "left"|"right", amount: 0-1 }
        if (target === "player" && this.player.raiseArm) {
          this.player.raiseArm(params.hand || "right", params.amount ?? 1);
          console.log(`🦾 Raised ${params.hand || "right"} arm`);
        }
        break;
        
      case "wave":
        // wave { hand: "left"|"right" }
        if (target === "player" && this.player.waveHand) {
          this.player.waveHand(params.hand || "right");
          console.log(`👋 Waving ${params.hand || "right"} hand`);
        }
        break;
        
      case "point_at":
        // point_at { position: {x,y,z}, hand: "left"|"right" }
        if (target === "player" && params.position && this.player.pointAtPosition) {
          this.player.pointAtPosition(params.position, params.hand || "right");
          console.log(`👆 Pointing at`, params.position);
        }
        break;
        
      case "look_at":
        // look_at { position: {x,y,z} }
        if (target === "player" && params.position && this.player.lookAtPosition) {
          this.player.lookAtPosition(params.position);
          console.log(`👀 Looking at`, params.position);
        }
        break;
        
      case "reach":
        // reach { position: {x,y,z}, hand: "left"|"right" }
        if (target === "player" && params.position && this.player.reachTo) {
          this.player.reachTo(params.position, params.hand || "right");
          console.log(`🤚 Reaching to`, params.position);
        }
        break;
        
      case "fingers":
        // fingers { hand: "left"|"right", pose: "open"|"fist"|"point"|"peace"|"thumbsUp"|"grab" }
        if (target === "player" && this.player.setFingers) {
          this.player.setFingers(params.hand || "right", params.pose || "open");
          console.log(`🖐️ Set ${params.hand || "right"} hand: ${params.pose}`);
        }
        break;
        
      case "motor_control":
        // motor_control { enabled: true|false }
        if (target === "player") {
          if (params.enabled) {
            this.player.enableMotorControl();
          } else {
            this.player.disableMotorControl();
          }
        }
        break;
        
      case "set_property": {
        const entity = this.registry.find(target);
        if (entity && typeof entity.applyProps === "function") {
          entity.applyProps(params);
        }
        break;
      }
      default:
        console.log("Unhandled command", command);
    }
  }
}
