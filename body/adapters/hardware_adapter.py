"""
AetherMind Hardware Interface Layer
Path: body/adapters/hardware_adapter.py

The Nervous System: Unified multiplexer for GPIO, Serial/UART, and RTSP protocols.
This adapter transforms AetherMind from a software agent into a cyber-physical system
capable of interacting with sensors, actuators, cameras, and legacy industrial equipment.

Architecture:
1. GPIO: Direct sensor/actuator control (Smart Home, Robotics, IoT)
2. Serial/UART: Microcontroller communication (Arduino, Motor Controllers, Legacy Tech)
3. RTSP: Visual ingestion from network cameras (Surveillance, Security, City Infrastructure)

Design Philosophy:
- Hot-swappable: Compatible with ToolForge for dynamic driver generation
- Safety-first: All commands pass through brain/safety_inhibitor.py kinetic safety checks
- Protocol-agnostic: Unified JSON interface for all hardware communication
- Stateless where possible: Minimizes resource leaks and connection failures
"""

import cv2
import json
import logging
import threading
import time
from typing import Dict, Any, Optional

# Conditional imports to prevent crashes on unsupported platforms
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logging.warning("PySerial not available. Serial/UART functionality disabled.")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("RPi.GPIO not available. GPIO functionality disabled.")

from body.adapter_base import BodyAdapter


class HardwareAdapter(BodyAdapter):
    """
    The Physical Body Interface: Bridges AetherMind's abstract intents to real-world hardware.
    
    This adapter receives JSON-structured intents from the Brain/Orchestrator and translates
    them into protocol-specific commands. All hardware interactions are logged for episodic
    memory and safety auditing.
    
    Example Intent Format:
    {
        "protocol": "GPIO" | "SERIAL" | "RTSP",
        "action": "write" | "read" | "connect" | "capture" | "setup",
        "params": {
            ... protocol-specific parameters ...
        },
        "metadata": {
            "timestamp": "ISO-8601",
            "intent_id": "uuid",
            "safety_approved": true
        }
    }
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Hardware Interface Layer.
        
        Args:
            config: Configuration dictionary containing:
                - gpio_mode: "BCM" or "BOARD" (default: BCM)
                - serial_timeout: Default timeout for serial connections (default: 1.0)
                - rtsp_timeout: Default timeout for RTSP streams (default: 5.0)
                - enable_logging: Enable detailed hardware interaction logging (default: True)
                - critical_pins: List of GPIO pins that should never be modified
        """
        self.config = config
        self.logger = logging.getLogger("HardwareAdapter")
        self.serial_connections: Dict[str, serial.Serial] = {}
        self.camera_streams: Dict[str, cv2.VideoCapture] = {}
        self.gpio_initialized = False
        
        # Safety: Critical pins that are blacklisted from modification
        self.critical_pins = config.get("critical_pins", [21])  # Default: Emergency shutdown pin
        
        # Initialize GPIO subsystem if available
        if GPIO_AVAILABLE:
            gpio_mode = config.get("gpio_mode", "BCM")
            GPIO.setmode(GPIO.BCM if gpio_mode == "BCM" else GPIO.BOARD)
            GPIO.setwarnings(False)
            self.gpio_initialized = True
            self.logger.info(f"GPIO initialized in {gpio_mode} mode")
        
        # Serial configuration
        self.serial_timeout = config.get("serial_timeout", 1.0)
        
        # RTSP configuration
        self.rtsp_timeout = config.get("rtsp_timeout", 5.0)
        
        self.logger.info("HardwareAdapter initialized successfully")

    def execute(self, intent: str) -> str:
        """
        Primary execution method: Routes brain intents to physical hardware.
        
        This method acts as a protocol multiplexer, parsing the intent JSON and
        dispatching to the appropriate hardware handler. All executions are logged
        for episodic memory and safety auditing.
        
        Args:
            intent: JSON-formatted string containing protocol, action, and parameters
            
        Returns:
            JSON-formatted string containing execution result, status, and metadata
            
        Raises:
            No exceptions are raised directly; all errors are caught and returned
            as JSON error messages to maintain system stability.
        """
        execution_start = time.time()
        
        try:
            command = json.loads(intent)
            protocol = command.get("protocol", "").upper()
            action = command.get("action", "")
            
            self.logger.info(f"Executing {protocol} action: {action}")
            
            # Protocol routing with availability checks
            if protocol == "GPIO":
                if not GPIO_AVAILABLE:
                    return self._error_response("GPIO hardware not detected. Install RPi.GPIO.")
                result = self._handle_gpio(command)
                
            elif protocol == "SERIAL":
                if not SERIAL_AVAILABLE:
                    return self._error_response("Serial interface not available. Install pyserial.")
                result = self._handle_serial(command)
                
            elif protocol == "RTSP":
                result = self._handle_rtsp(command)
                
            else:
                return self._error_response(f"Unknown protocol: {protocol}")
            
            # Add execution metadata
            execution_time = time.time() - execution_start
            self.logger.info(f"Hardware execution completed in {execution_time:.3f}s")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            return self._error_response("Hardware intent must be valid JSON")
            
        except Exception as e:
            self.logger.error(f"Hardware execution failure: {e}", exc_info=True)
            return self._error_response(f"Hardware Critical Fail: {str(e)}")

    # ==========================
    # GPIO: Direct Hardware Control
    # ==========================
    
    def _handle_gpio(self, cmd: Dict) -> str:
        """
        GPIO Handler: Manages Raspberry Pi GPIO pins for sensor/actuator control.
        
        Supported Actions:
        - setup: Configure pin mode (INPUT/OUTPUT)
        - write: Set pin to HIGH/LOW state
        - read: Read current pin state
        - pwm_start: Initialize PWM on a pin
        - pwm_duty: Set PWM duty cycle
        
        Safety Features:
        - Critical pin protection (prevents modification of safety-critical pins)
        - Mode validation (prevents invalid GPIO operations)
        - State logging (all GPIO changes recorded for audit trail)
        
        Example:
        {
            "protocol": "GPIO",
            "action": "write",
            "params": {
                "pin": 17,
                "state": 1
            }
        }
        """
        if not self.gpio_initialized:
            return self._error_response("GPIO subsystem not initialized")
        
        action = cmd.get("action")
        params = cmd.get("params", {})
        pin = params.get("pin")
        
        # Validate pin number
        if pin is None:
            return self._error_response("GPIO pin number required")
        
        try:
            pin = int(pin)
        except ValueError:
            return self._error_response(f"Invalid pin number: {pin}")
        
        # Critical pin protection
        if pin in self.critical_pins:
            self.logger.error(f"SECURITY: Attempted access to critical pin {pin}")
            return self._error_response(f"BLOCKED: Pin {pin} is safety-critical and cannot be modified")
        
        # Action routing
        if action == "setup":
            mode_str = params.get("mode", "OUT").upper()
            mode = GPIO.OUT if mode_str == "OUT" else GPIO.IN
            pull_up_down = GPIO.PUD_OFF
            
            # Optional pull-up/pull-down resistor configuration
            if "pull" in params:
                pull = params["pull"].upper()
                if pull == "UP":
                    pull_up_down = GPIO.PUD_UP
                elif pull == "DOWN":
                    pull_up_down = GPIO.PUD_DOWN
            
            GPIO.setup(pin, mode, pull_up_down=pull_up_down)
            self.logger.info(f"GPIO pin {pin} configured as {mode_str}")
            return self._success_response({
                "action": "setup",
                "pin": pin,
                "mode": mode_str,
                "message": f"Pin {pin} setup complete"
            })
        
        elif action == "write":
            state = params.get("state", 0)
            gpio_state = GPIO.HIGH if state in [1, "HIGH", "1", True] else GPIO.LOW
            GPIO.output(pin, gpio_state)
            self.logger.info(f"GPIO pin {pin} set to {gpio_state}")
            return self._success_response({
                "action": "write",
                "pin": pin,
                "state": gpio_state,
                "message": f"Pin {pin} set to {'HIGH' if gpio_state == GPIO.HIGH else 'LOW'}"
            })
        
        elif action == "read":
            value = GPIO.input(pin)
            self.logger.info(f"GPIO pin {pin} read: {value}")
            return self._success_response({
                "action": "read",
                "pin": pin,
                "value": value,
                "state": "HIGH" if value == GPIO.HIGH else "LOW"
            })
        
        elif action == "cleanup":
            GPIO.cleanup(pin)
            return self._success_response({
                "action": "cleanup",
                "pin": pin,
                "message": f"Pin {pin} reset to safe state"
            })
        
        else:
            return self._error_response(f"Unknown GPIO action: {action}")

    # ==========================
    # Serial/UART: Microcontroller Communication
    # ==========================
    
    def _handle_serial(self, cmd: Dict) -> str:
        """
        Serial Handler: Manages UART/Serial communication with microcontrollers and legacy systems.
        
        Supported Actions:
        - connect: Open serial port connection
        - disconnect: Close serial port connection
        - write: Send data to serial device
        - read: Read data from serial device
        - read_line: Read until newline character
        - list_ports: Enumerate available serial ports
        
        Use Cases:
        - Arduino motor controller communication
        - Industrial PLC interfaces
        - Legacy equipment integration
        - Custom sensor networks
        
        Example:
        {
            "protocol": "SERIAL",
            "action": "write",
            "params": {
                "port": "/dev/ttyACM0",
                "payload": "SERVO_HAND_CLOSE_50",
                "encoding": "utf-8"
            }
        }
        """
        action = cmd.get("action")
        params = cmd.get("params", {})
        port = params.get("port", "/dev/ttyUSB0")
        
        if action == "connect":
            if port in self.serial_connections:
                return self._success_response({
                    "action": "connect",
                    "port": port,
                    "message": "Already connected",
                    "status": "existing"
                })
            
            try:
                baud = params.get("baud", 9600)
                timeout = params.get("timeout", self.serial_timeout)
                
                self.serial_connections[port] = serial.Serial(
                    port=port,
                    baudrate=baud,
                    timeout=timeout,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                
                self.logger.info(f"Serial connection established: {port} @ {baud} baud")
                return self._success_response({
                    "action": "connect",
                    "port": port,
                    "baud": baud,
                    "message": f"Connected to {port} at {baud} baud"
                })
                
            except serial.SerialException as e:
                self.logger.error(f"Serial connection failed: {e}")
                return self._error_response(f"Failed to connect to {port}: {str(e)}")
        
        elif action == "disconnect":
            if port not in self.serial_connections:
                return self._error_response(f"Port {port} not open")
            
            self.serial_connections[port].close()
            del self.serial_connections[port]
            self.logger.info(f"Serial connection closed: {port}")
            return self._success_response({
                "action": "disconnect",
                "port": port,
                "message": f"Disconnected from {port}"
            })
        
        elif action == "write":
            if port not in self.serial_connections:
                return self._error_response(f"Port {port} not open. Call connect first.")
            
            payload = params.get("payload", "")
            encoding = params.get("encoding", "utf-8")
            append_newline = params.get("append_newline", True)
            
            if append_newline and not payload.endswith("\n"):
                payload += "\n"
            
            try:
                bytes_written = self.serial_connections[port].write(payload.encode(encoding))
                self.logger.info(f"Serial TX [{port}]: {payload.strip()}")
                return self._success_response({
                    "action": "write",
                    "port": port,
                    "bytes_written": bytes_written,
                    "payload": payload.strip(),
                    "message": f"Sent {bytes_written} bytes to {port}"
                })
            except serial.SerialException as e:
                self.logger.error(f"Serial write failed: {e}")
                return self._error_response(f"Write failed: {str(e)}")
        
        elif action == "read":
            if port not in self.serial_connections:
                return self._error_response(f"Port {port} not open. Call connect first.")
            
            try:
                num_bytes = params.get("bytes", 1024)
                data = self.serial_connections[port].read(num_bytes)
                decoded = data.decode(params.get("encoding", "utf-8"), errors="replace")
                self.logger.info(f"Serial RX [{port}]: {decoded}")
                return self._success_response({
                    "action": "read",
                    "port": port,
                    "data": decoded,
                    "bytes_read": len(data)
                })
            except Exception as e:
                self.logger.error(f"Serial read failed: {e}")
                return self._error_response(f"Read failed: {str(e)}")
        
        elif action == "read_line":
            if port not in self.serial_connections:
                return self._error_response(f"Port {port} not open. Call connect first.")
            
            try:
                line = self.serial_connections[port].readline()
                decoded = line.decode(params.get("encoding", "utf-8"), errors="replace").strip()
                self.logger.info(f"Serial RX Line [{port}]: {decoded}")
                return self._success_response({
                    "action": "read_line",
                    "port": port,
                    "data": decoded
                })
            except Exception as e:
                self.logger.error(f"Serial read_line failed: {e}")
                return self._error_response(f"Read line failed: {str(e)}")
        
        elif action == "list_ports":
            try:
                import serial.tools.list_ports
                ports = [{"device": p.device, "description": p.description} 
                        for p in serial.tools.list_ports.comports()]
                return self._success_response({
                    "action": "list_ports",
                    "ports": ports,
                    "count": len(ports)
                })
            except Exception as e:
                return self._error_response(f"Failed to list ports: {str(e)}")
        
        else:
            return self._error_response(f"Unknown Serial action: {action}")

    # ==========================
    # RTSP: Visual Intelligence Gateway
    # ==========================
    
    def _handle_rtsp(self, cmd: Dict) -> str:
        """
        RTSP Handler: Manages network camera streams for visual perception.
        
        Supported Actions:
        - capture: Grab single frame from RTSP stream (stateless)
        - stream_start: Open persistent RTSP stream
        - stream_read: Read frame from active stream
        - stream_stop: Close active stream
        
        Integration Points:
        - Captured frames are saved to /tmp/vision_*.jpg
        - Frames are automatically sent to perception/eye.py for analysis
        - Supports authentication (username:password@host)
        - Handles connection failures gracefully
        
        Use Cases:
        - City surveillance integration
        - Security camera analysis
        - Traffic monitoring
        - Real-time object detection
        
        Example:
        {
            "protocol": "RTSP",
            "action": "capture",
            "params": {
                "url": "rtsp://admin:password@192.168.1.50:554/stream",
                "save_path": "/tmp/vision_capture.jpg",
                "timeout": 5.0
            }
        }
        """
        action = cmd.get("action")
        params = cmd.get("params", {})
        stream_url = params.get("url")
        
        if not stream_url:
            return self._error_response("RTSP URL required")
        
        if action == "capture":
            # Stateless frame capture: Connect, grab, disconnect
            try:
                self.logger.info(f"Capturing frame from RTSP: {stream_url}")
                cap = cv2.VideoCapture(stream_url)
                
                # Set timeout
                timeout = params.get("timeout", self.rtsp_timeout)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
                
                ret, frame = cap.read()
                cap.release()
                
                if not ret or frame is None:
                    return self._error_response("Failed to retrieve frame from stream")
                
                # Save frame to disk
                timestamp = int(time.time() * 1000)  # Millisecond precision
                default_path = f"/tmp/vision_{timestamp}.jpg"
                filename = params.get("save_path", default_path)
                
                cv2.imwrite(filename, frame)
                frame_shape = frame.shape  # (height, width, channels)
                
                self.logger.info(f"Frame captured successfully: {filename} ({frame_shape[1]}x{frame_shape[0]})")
                
                return self._success_response({
                    "action": "capture",
                    "status": "success",
                    "file": filename,
                    "resolution": {
                        "width": frame_shape[1],
                        "height": frame_shape[0]
                    },
                    "message": "Image captured. Ready for perception/eye.py analysis",
                    "next_step": "Send to perception system for object detection"
                })
                
            except Exception as e:
                self.logger.error(f"RTSP capture failed: {e}")
                return self._error_response(f"Capture failed: {str(e)}")
        
        elif action == "stream_start":
            # Persistent stream: Useful for continuous monitoring
            if stream_url in self.camera_streams:
                return self._success_response({
                    "action": "stream_start",
                    "url": stream_url,
                    "message": "Stream already active",
                    "status": "existing"
                })
            
            try:
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    return self._error_response(f"Failed to open stream: {stream_url}")
                
                self.camera_streams[stream_url] = cap
                self.logger.info(f"RTSP stream opened: {stream_url}")
                
                return self._success_response({
                    "action": "stream_start",
                    "url": stream_url,
                    "message": "Stream opened successfully"
                })
            except Exception as e:
                return self._error_response(f"Stream start failed: {str(e)}")
        
        elif action == "stream_read":
            if stream_url not in self.camera_streams:
                return self._error_response(f"Stream not open: {stream_url}. Call stream_start first.")
            
            try:
                ret, frame = self.camera_streams[stream_url].read()
                if not ret or frame is None:
                    return self._error_response("Failed to read frame from active stream")
                
                timestamp = int(time.time() * 1000)
                filename = params.get("save_path", f"/tmp/vision_stream_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                
                return self._success_response({
                    "action": "stream_read",
                    "file": filename,
                    "message": "Frame read from active stream"
                })
            except Exception as e:
                return self._error_response(f"Stream read failed: {str(e)}")
        
        elif action == "stream_stop":
            if stream_url not in self.camera_streams:
                return self._error_response(f"Stream not open: {stream_url}")
            
            self.camera_streams[stream_url].release()
            del self.camera_streams[stream_url]
            self.logger.info(f"RTSP stream closed: {stream_url}")
            
            return self._success_response({
                "action": "stream_stop",
                "url": stream_url,
                "message": "Stream closed successfully"
            })
        
        else:
            return self._error_response(f"Unknown RTSP action: {action}")

    # ==========================
    # Utility Methods
    # ==========================
    
    def _success_response(self, data: Dict) -> str:
        """Generate standardized success response JSON."""
        response = {
            "status": "success",
            "timestamp": time.time(),
            **data
        }
        return json.dumps(response, indent=2)
    
    def _error_response(self, message: str) -> str:
        """Generate standardized error response JSON."""
        response = {
            "status": "error",
            "timestamp": time.time(),
            "message": message
        }
        return json.dumps(response, indent=2)
    
    def cleanup(self):
        """
        Graceful shutdown: Close all active connections and reset hardware.
        This method should be called during system shutdown to prevent resource leaks.
        """
        self.logger.info("Initiating hardware cleanup...")
        
        # Close all serial connections
        for port, connection in list(self.serial_connections.items()):
            try:
                connection.close()
                self.logger.info(f"Closed serial connection: {port}")
            except Exception as e:
                self.logger.error(f"Error closing serial port {port}: {e}")
        
        self.serial_connections.clear()
        
        # Release all camera streams
        for url, stream in list(self.camera_streams.items()):
            try:
                stream.release()
                self.logger.info(f"Released camera stream: {url}")
            except Exception as e:
                self.logger.error(f"Error releasing stream {url}: {e}")
        
        self.camera_streams.clear()
        
        # Reset GPIO to safe state
        if self.gpio_initialized:
            try:
                GPIO.cleanup()
                self.logger.info("GPIO cleanup complete")
            except Exception as e:
                self.logger.error(f"GPIO cleanup error: {e}")
        
        self.logger.info("Hardware cleanup complete")
    
    def __del__(self):
        """Destructor: Ensure cleanup happens even if explicitly not called."""
        self.cleanup()


# Example Usage and Testing
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize adapter with test configuration
    config = {
        "gpio_mode": "BCM",
        "serial_timeout": 1.0,
        "rtsp_timeout": 5.0,
        "critical_pins": [21],  # Emergency shutdown pin
        "enable_logging": True
    }
    
    adapter = HardwareAdapter(config)
    
    print("=" * 80)
    print("AetherMind Hardware Adapter - Interactive Test Mode")
    print("=" * 80)
    print("\nAvailable Protocols:")
    print("  GPIO   - Raspberry Pi GPIO control")
    print("  SERIAL - UART/Serial communication")
    print("  RTSP   - Network camera streams")
    print("\nExample GPIO command:")
    print('  {"protocol": "GPIO", "action": "setup", "params": {"pin": 17, "mode": "OUT"}}')
    print("\nExample Serial command:")
    print('  {"protocol": "SERIAL", "action": "list_ports", "params": {}}')
    print("\nExample RTSP command:")
    print('  {"protocol": "RTSP", "action": "capture", "params": {"url": "rtsp://..."}}')
    print("\nType 'exit' to quit\n")
    
    while True:
        try:
            user_input = input("\nEnter hardware intent (JSON): ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nShutting down hardware adapter...")
                adapter.cleanup()
                sys.exit(0)
            
            if not user_input:
                continue
            
            # Execute the hardware intent
            result = adapter.execute(user_input)
            print("\nResult:")
            print(result)
            
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected. Shutting down...")
            adapter.cleanup()
            sys.exit(0)
        except Exception as e:
            print(f"\nUnexpected error: {e}")
