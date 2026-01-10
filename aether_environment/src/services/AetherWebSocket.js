/**
 * AetherWebSocket.js
 * 
 * WebSocket client for real-time bidirectional communication with backend.
 * Receives control commands from the Brain and sends perception updates.
 */

const WS_URL = import.meta.env.VITE_AETHER_WS_URL ?? "ws://localhost:8000/ws/embodiment";

export class AetherWebSocket {
  constructor(options = {}) {
    this.options = {
      url: WS_URL,
      autoReconnect: true,
      reconnectInterval: 3000,
      maxReconnectAttempts: 10,
      ...options
    };
    
    this.ws = null;
    this.connected = false;
    this.reconnectAttempts = 0;
    this.reconnectTimeout = null;
    
    // Callbacks
    this.onCommand = null;  // Called when command received from backend
    this.onConnected = null;
    this.onDisconnected = null;
    this.onError = null;
    
    console.log("🔌 AetherWebSocket created");
  }
  
  /**
   * Connect to WebSocket server
   */
  connect() {
    if (this.ws && this.connected) {
      console.warn("Already connected to WebSocket");
      return;
    }
    
    console.log(`🔌 Connecting to ${this.options.url}...`);
    
    try {
      this.ws = new WebSocket(this.options.url);
      
      this.ws.onopen = () => {
        console.log("✅ WebSocket connected");
        this.connected = true;
        this.reconnectAttempts = 0;
        
        if (this.onConnected) {
          this.onConnected();
        }
      };
      
      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };
      
      this.ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
        if (this.onError) {
          this.onError(error);
        }
      };
      
      this.ws.onclose = () => {
        console.log("🔌 WebSocket disconnected");
        this.connected = false;
        
        if (this.onDisconnected) {
          this.onDisconnected();
        }
        
        // Auto-reconnect
        if (this.options.autoReconnect && 
            this.reconnectAttempts < this.options.maxReconnectAttempts) {
          this.reconnectAttempts++;
          console.log(`🔄 Reconnecting in ${this.options.reconnectInterval}ms (attempt ${this.reconnectAttempts}/${this.options.maxReconnectAttempts})`);
          
          this.reconnectTimeout = setTimeout(() => {
            this.connect();
          }, this.options.reconnectInterval);
        }
      };
      
    } catch (error) {
      console.error("Failed to create WebSocket:", error);
      if (this.onError) {
        this.onError(error);
      }
    }
  }
  
  /**
   * Handle incoming message from backend
   */
  handleMessage(data) {
    try {
      const message = JSON.parse(data);
      
      // Check message type
      if (message.type === "ack") {
        // Acknowledgment - ignore
        return;
      }
      
      // Command from backend - execute it
      if (this.onCommand) {
        this.onCommand(message);
      }
      
    } catch (error) {
      console.error("Failed to parse WebSocket message:", error);
    }
  }
  
  /**
   * Send perception update to backend
   */
  sendPerceptionUpdate(perceptionData) {
    if (!this.connected || !this.ws) {
      console.warn("Cannot send perception - not connected");
      return false;
    }
    
    try {
      const message = {
        type: "perception_update",
        data: perceptionData,
        timestamp: Date.now()
      };
      
      this.ws.send(JSON.stringify(message));
      return true;
      
    } catch (error) {
      console.error("Failed to send perception update:", error);
      return false;
    }
  }
  
  /**
   * Send custom message to backend
   */
  send(message) {
    if (!this.connected || !this.ws) {
      console.warn("Cannot send message - not connected");
      return false;
    }
    
    try {
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error("Failed to send message:", error);
      return false;
    }
  }
  
  /**
   * Disconnect from WebSocket
   */
  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.connected = false;
    console.log("🔌 WebSocket disconnected");
  }
  
  /**
   * Check if connected
   */
  isConnected() {
    return this.connected;
  }
}

// Singleton instance
let websocketInstance = null;

export function getAetherWebSocket() {
  if (!websocketInstance) {
    websocketInstance = new AetherWebSocket();
  }
  return websocketInstance;
}

export function initializeWebSocket() {
  const ws = getAetherWebSocket();
  ws.connect();
  return ws;
}
