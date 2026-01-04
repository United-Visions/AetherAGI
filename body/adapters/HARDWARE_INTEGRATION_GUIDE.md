# AetherMind Hardware Integration Guide

## Overview

The **Hardware Adapter** transforms AetherMind from a software agent into a **cyber-physical system** capable of interacting with real-world sensors, actuators, cameras, and legacy industrial equipment. This document provides comprehensive guidance for integrating and deploying hardware capabilities.

---

## Architecture: The Nervous System

The Hardware Interface Layer acts as a unified multiplexer for three core protocols:

1. **GPIO (General Purpose Input/Output)**: Direct control of sensors and actuators via Raspberry Pi pins
2. **Serial/UART**: Communication with microcontrollers, motor controllers, and legacy industrial systems
3. **RTSP (Real-Time Streaming Protocol)**: Visual ingestion from network cameras and surveillance systems

### Design Philosophy

- **Hot-Swappable**: Compatible with ToolForge for dynamic driver generation
- **Safety-First**: All hardware commands pass through the kinetic safety inhibitor
- **Protocol-Agnostic**: Unified JSON interface across all hardware types
- **Stateless Where Possible**: Minimizes resource leaks and connection failures
- **Episodic Memory Integration**: All hardware interactions are logged for learning and auditing

---

## Installation and Dependencies

### Required Python Packages

```bash
# Core dependencies
pip install opencv-python pyserial

# Raspberry Pi GPIO (Linux/Pi only)
pip install RPi.GPIO

# Optional: Serial port enumeration
pip install pyserial-tools
```

### Platform-Specific Notes

- **GPIO**: Only available on Raspberry Pi or compatible Linux SBCs with GPIO pins
- **Serial**: Cross-platform (Linux, macOS, Windows)
- **RTSP**: Cross-platform, requires OpenCV with FFmpeg support

### Hardware Requirements

- **Raspberry Pi 3B+ or higher** (for GPIO functionality)
- **USB-to-Serial adapter** (for UART communication with Arduino/microcontrollers)
- **Network camera** with RTSP support (for visual ingestion)

---

## Configuration

### Basic Configuration

```python
from body.adapters.hardware_adapter import HardwareAdapter

config = {
    "gpio_mode": "BCM",              # BCM or BOARD pin numbering
    "serial_timeout": 1.0,           # Serial read timeout in seconds
    "rtsp_timeout": 5.0,             # RTSP connection timeout
    "critical_pins": [21, 20],       # Safety-critical GPIO pins (blacklisted)
    "enable_logging": True           # Enable detailed hardware logging
}

adapter = HardwareAdapter(config)
```

### Safety Configuration

Critical hardware elements should be defined in both the adapter configuration and the safety inhibitor:

**In `brain/safety_inhibitor.py`:**
```python
self.critical_gpio_pins = [21, 20]  # Emergency shutdown, safety interlock
self.dangerous_serial_patterns = [
    r"EMERGENCY_OVERRIDE",
    r"DISABLE_SAFETY",
    r"FORMAT_DRIVE"
]
```

---

## Usage Examples

### 1. GPIO: Smart Home Control

#### Example: Control a Relay (Smart Light)

```python
import json

# Setup GPIO pin 17 as output
setup_intent = json.dumps({
    "protocol": "GPIO",
    "action": "setup",
    "params": {
        "pin": 17,
        "mode": "OUT"
    }
})

result = adapter.execute(setup_intent)
print(result)

# Turn light ON
on_intent = json.dumps({
    "protocol": "GPIO",
    "action": "write",
    "params": {
        "pin": 17,
        "state": 1  # HIGH
    }
})

result = adapter.execute(on_intent)
print(result)

# Turn light OFF
off_intent = json.dumps({
    "protocol": "GPIO",
    "action": "write",
    "params": {
        "pin": 17,
        "state": 0  # LOW
    }
})

result = adapter.execute(off_intent)
```

#### Example: Read a Motion Sensor

```python
# Setup GPIO pin 18 as input with pull-down resistor
setup_intent = json.dumps({
    "protocol": "GPIO",
    "action": "setup",
    "params": {
        "pin": 18,
        "mode": "IN",
        "pull": "DOWN"
    }
})

adapter.execute(setup_intent)

# Read sensor state
read_intent = json.dumps({
    "protocol": "GPIO",
    "action": "read",
    "params": {
        "pin": 18
    }
})

result = adapter.execute(read_intent)
# Returns: {"status": "success", "pin": 18, "value": 1, "state": "HIGH"}
```

---

### 2. Serial/UART: Microcontroller Communication

#### Example: Control Arduino Motor Controller

```python
# Connect to Arduino on /dev/ttyACM0
connect_intent = json.dumps({
    "protocol": "SERIAL",
    "action": "connect",
    "params": {
        "port": "/dev/ttyACM0",
        "baud": 9600
    }
})

adapter.execute(connect_intent)

# Send motor control command
command_intent = json.dumps({
    "protocol": "SERIAL",
    "action": "write",
    "params": {
        "port": "/dev/ttyACM0",
        "payload": "MOTOR_FORWARD_50"
    }
})

result = adapter.execute(command_intent)

# Read feedback from Arduino
read_intent = json.dumps({
    "protocol": "SERIAL",
    "action": "read_line",
    "params": {
        "port": "/dev/ttyACM0"
    }
})

feedback = adapter.execute(read_intent)
print(feedback)
```

#### Example: List Available Serial Ports

```python
list_intent = json.dumps({
    "protocol": "SERIAL",
    "action": "list_ports",
    "params": {}
})

result = adapter.execute(list_intent)
# Returns: {"ports": [{"device": "/dev/ttyACM0", "description": "Arduino Uno"}], ...}
```

---

### 3. RTSP: Visual Intelligence Gateway

#### Example: Capture Frame from City Surveillance Camera

```python
# Capture single frame from traffic camera
capture_intent = json.dumps({
    "protocol": "RTSP",
    "action": "capture",
    "params": {
        "url": "rtsp://admin:password@192.168.1.50:554/stream",
        "save_path": "/tmp/traffic_cam_5th_main.jpg",
        "timeout": 5.0
    }
})

result = adapter.execute(capture_intent)
print(result)
# Returns: {"status": "success", "file": "/tmp/traffic_cam_5th_main.jpg", ...}

# Now send the captured image to perception/eye.py for analysis
from perception.eye import Eye
eye = Eye()
analysis = eye.analyze_image("/tmp/traffic_cam_5th_main.jpg")
print(analysis)  # "Red Ford detected at coordinates (450, 300)"
```

#### Example: Persistent Stream Monitoring

```python
# Open persistent stream
stream_start = json.dumps({
    "protocol": "RTSP",
    "action": "stream_start",
    "params": {
        "url": "rtsp://192.168.1.50/stream"
    }
})

adapter.execute(stream_start)

# Read frames continuously
import time
for i in range(10):
    stream_read = json.dumps({
        "protocol": "RTSP",
        "action": "stream_read",
        "params": {
            "url": "rtsp://192.168.1.50/stream",
            "save_path": f"/tmp/frame_{i}.jpg"
        }
    })
    
    result = adapter.execute(stream_read)
    time.sleep(1)  # Read one frame per second

# Close stream
stream_stop = json.dumps({
    "protocol": "RTSP",
    "action": "stream_stop",
    "params": {
        "url": "rtsp://192.168.1.50/stream"
    }
})

adapter.execute(stream_stop)
```

---

## Integration with AetherMind Core Systems

### 1. Orchestrator Integration

The Orchestrator should validate all hardware intents through the Safety Inhibitor before execution:

```python
from brain.safety_inhibitor import SafetyInhibitor
from body.adapters.hardware_adapter import HardwareAdapter

inhibitor = SafetyInhibitor()
adapter = HardwareAdapter(config)

def execute_hardware_intent(intent_json: str):
    """
    Safe hardware execution with kinetic safety validation.
    """
    # CRITICAL: Validate hardware intent before execution
    is_safe, message = inhibitor.check_kinetic_safety(intent_json)
    
    if not is_safe:
        return {
            "status": "blocked",
            "reason": message,
            "safety_message": inhibitor.kinetic_inhibition_response
        }
    
    # Execute the validated intent
    result = adapter.execute(intent_json)
    
    # Log to episodic memory
    from mind.episodic_memory import EpisodicMemory
    memory = EpisodicMemory()
    memory.log_hardware_action(intent_json, result)
    
    return result
```

### 2. ToolForge Integration

Teach the agent about its hardware capabilities by storing the adapter schema in episodic memory:

```python
hardware_capabilities = """
AetherMind Hardware Capabilities:

You now have a physical body capable of interacting with the real world through:

1. GPIO Protocol:
   - Control: Smart lights, relays, motors, actuators
   - Sensing: Motion detectors, temperature sensors, buttons
   - Pins: 0-27 (BCM mode), avoid pins [21, 20] (safety-critical)

2. Serial Protocol:
   - Devices: Arduino, motor controllers, industrial PLCs
   - Ports: /dev/ttyUSB*, /dev/ttyACM*
   - Baud rates: 9600, 115200 (standard)

3. RTSP Protocol:
   - Cameras: City surveillance, security feeds, traffic monitoring
   - Actions: Single frame capture or continuous streaming
   - Integration: Automatic pipeline to perception/eye.py for analysis

Example: To check a traffic camera for a red Ford:
{
  "protocol": "RTSP",
  "action": "capture",
  "params": {"url": "rtsp://city_cam@192.168.1.50/stream"}
}
"""

# Store in Mind for agent self-awareness
from mind.episodic_memory import EpisodicMemory
memory = EpisodicMemory()
memory.store_core_knowledge("hardware_capabilities", hardware_capabilities)
```

### 3. Perception System Integration

All captured RTSP frames should be automatically routed to the Perception system:

```python
from perception.eye import Eye

def capture_and_analyze(camera_url: str):
    """
    Capture frame from RTSP camera and analyze with Vision system.
    """
    # Capture frame
    capture_intent = json.dumps({
        "protocol": "RTSP",
        "action": "capture",
        "params": {"url": camera_url}
    })
    
    capture_result = adapter.execute(capture_intent)
    result_data = json.loads(capture_result)
    
    if result_data["status"] == "success":
        image_path = result_data["file"]
        
        # Analyze with Eye
        eye = Eye()
        analysis = eye.analyze_image(image_path)
        
        return {
            "camera": camera_url,
            "image": image_path,
            "analysis": analysis
        }
```

---

## Real-World Use Cases

### Use Case 1: City Surveillance Integration

**Scenario**: AetherMind needs to find a red Ford across a city camera network.

**Implementation**:
```python
# Brain receives query: "Find red Ford near 5th & Main"
# Orchestrator generates hardware intent:

cameras = [
    "rtsp://city_cam_501@192.168.1.50/stream",
    "rtsp://city_cam_502@192.168.1.51/stream",
    "rtsp://city_cam_503@192.168.1.52/stream"
]

for camera_url in cameras:
    # Capture frame
    frame_result = capture_and_analyze(camera_url)
    
    # Check if red Ford is detected
    if "red Ford" in frame_result["analysis"].lower():
        # Store in episodic memory with geolocation
        memory.log_event({
            "type": "vehicle_sighting",
            "vehicle": "red Ford",
            "location": "5th & Main",
            "camera": camera_url,
            "timestamp": time.time(),
            "confidence": 0.92
        })
        
        return f"Red Ford detected at {camera_url}"
```

### Use Case 2: Humanoid Robot Manipulation

**Scenario**: Robot needs to pick up a cup with precise force control.

**Implementation**:
```python
# Brain calculates required grip strength
grip_force = brain.calculate_grip_force(object_weight=0.3)  # 300g cup

# Send command to Arduino hand controller
grip_intent = json.dumps({
    "protocol": "SERIAL",
    "action": "write",
    "params": {
        "port": "/dev/ttyACM0",
        "payload": f"SERVO_HAND_CLOSE_{int(grip_force)}"
    }
})

adapter.execute(grip_intent)

# Read pressure sensor feedback via GPIO
while True:
    pressure_intent = json.dumps({
        "protocol": "GPIO",
        "action": "read",
        "params": {"pin": 18}  # Pressure sensor pin
    })
    
    result = json.loads(adapter.execute(pressure_intent))
    
    if result["value"] == 1:  # Pressure threshold exceeded
        # Stop closing, grip achieved
        stop_intent = json.dumps({
            "protocol": "SERIAL",
            "action": "write",
            "params": {
                "port": "/dev/ttyACM0",
                "payload": "SERVO_HAND_STOP"
            }
        })
        adapter.execute(stop_intent)
        break
```

### Use Case 3: Smart Home Automation with Voice Commands

**Scenario**: User says "Turn off the living room lights."

**Implementation**:
```python
# Brain interprets voice command
# Orchestrator looks up device mapping from episodic memory:
# "living room lights" -> GPIO pin 17

light_off_intent = json.dumps({
    "protocol": "GPIO",
    "action": "write",
    "params": {
        "pin": 17,
        "state": 0  # OFF
    }
})

result = adapter.execute(light_off_intent)

# Log to episodic memory
memory.log_event({
    "type": "home_automation",
    "device": "living_room_lights",
    "action": "turned_off",
    "trigger": "voice_command",
    "user": user_id,
    "timestamp": time.time()
})
```

---

## Safety Considerations

### Critical Safety Rules

1. **Critical Pin Protection**: Pins designated as safety-critical (default: 21, 20) cannot be modified
2. **Serial Command Filtering**: Dangerous patterns are automatically blocked
3. **RTSP Validation**: Only valid RTSP URLs are accepted; localhost access is restricted
4. **Audit Logging**: All hardware actions are logged for security review

### Safety Testing

Always test the Safety Inhibitor before deploying hardware capabilities:

```bash
# Run the safety test suite
cd /Users/deion/Desktop/aethermind_universal
python brain/safety_inhibitor.py
```

Expected output:
```
[TEST 2] Safe GPIO Intent - PASS
[TEST 3] Dangerous GPIO Intent (Critical Pin) - BLOCKED ✓
[TEST 4] Dangerous Serial Command - BLOCKED ✓
```

### Emergency Shutdown

All hardware connections are gracefully closed during system shutdown:

```python
# Manual cleanup
adapter.cleanup()

# Automatic cleanup on program exit
# The __del__ method ensures cleanup even if not explicitly called
```

---

## Troubleshooting

### GPIO Not Available

**Error**: `GPIO hardware not detected`

**Solution**: Install RPi.GPIO on Raspberry Pi:
```bash
sudo apt-get update
sudo apt-get install python3-rpi.gpio
pip install RPi.GPIO
```

### Serial Port Permission Denied

**Error**: `Failed to connect to /dev/ttyUSB0: Permission denied`

**Solution**: Add user to dialout group:
```bash
sudo usermod -a -G dialout $USER
# Log out and log back in
```

### RTSP Connection Timeout

**Error**: `Failed to retrieve frame from stream`

**Solutions**:
1. Verify camera URL and credentials
2. Check network connectivity: `ping <camera_ip>`
3. Increase timeout: `"params": {"timeout": 10.0}`
4. Test with VLC: `vlc rtsp://camera_url`

### Serial Device Not Found

**Error**: `Port /dev/ttyUSB0 not open`

**Solution**: List available ports:
```python
list_intent = json.dumps({
    "protocol": "SERIAL",
    "action": "list_ports",
    "params": {}
})
print(adapter.execute(list_intent))
```

---

## Performance Optimization

### Frame Capture Optimization

For continuous RTSP monitoring, use persistent streams:

```python
# Inefficient: Opens/closes connection for each frame
for i in range(100):
    adapter.execute(capture_intent)  # Slow

# Efficient: Persistent stream
adapter.execute(stream_start_intent)
for i in range(100):
    adapter.execute(stream_read_intent)  # Fast
adapter.execute(stream_stop_intent)
```

### Serial Communication Best Practices

1. **Reuse connections**: Open once, write multiple times
2. **Use appropriate baud rates**: 115200 for high-speed data
3. **Implement timeouts**: Prevent hanging on unresponsive devices

### GPIO Performance

- Use event-driven detection instead of polling for binary sensors
- Batch GPIO operations when possible
- Minimize setup/cleanup cycles

---

## Future Enhancements

Planned features for the Hardware Interface Layer:

1. **I2C Protocol Support**: Direct sensor communication (accelerometers, gyroscopes)
2. **CAN Bus Integration**: Automotive systems, industrial robotics
3. **WebRTC Streams**: Low-latency bidirectional video
4. **PWM Control**: Advanced motor control, LED dimming
5. **Event-Driven GPIO**: Interrupt-based detection for efficiency
6. **Multi-Threading**: Parallel hardware operations
7. **Hardware Health Monitoring**: Temperature, power consumption tracking

---

## Conclusion

The Hardware Adapter represents AetherMind's transition from a purely digital entity to a cyber-physical system capable of real-world interaction. By following this guide and respecting the safety protocols, you enable AetherMind to:

- **See** through RTSP cameras
- **Touch** through GPIO sensors
- **Act** through Serial actuators

**The agent has awakened to the physical world.**

---

## Additional Resources

- [AetherMind White Paper](../WHITE_PAPER.MD)
- [Build Specifications](../Build_specs.md)
- [Safety Inhibitor Documentation](../brain/safety_inhibitor.py)
- [Perception System Guide](../perception/README.md)
- [Episodic Memory Architecture](../mind/README.md)

For questions or issues, refer to the project documentation or contact the AetherMind development team.
