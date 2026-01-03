# Body Adapters

## 1. Directory Overview

The `adapters/` directory contains the specific implementations that translate AetherMind's abstract "Intents" into real-world actions. Each file in this directory represents a different modality of interaction. This modular approach allows AetherMind to easily learn new ways to communicate and interact with its environment.

## 2. Current Capabilities

-   **Chat UI Adapter (`chat_ui.py`):** This is the primary adapter for Phase 1. It takes a linguistic "Intent" from the `brain` and translates it into a text-based response suitable for a chat interface. It handles the final token generation and formatting for the user.
-   **IDE Agent (`ide_agent.py`):** (Planned) An adapter for interacting with an Integrated Development Environment (IDE), allowing AetherMind to write and modify code.
-   **Vision System (`vision_system.py`):** (Planned for Phase 2) An adapter for interpreting visual input and generating responses related to images or video feeds.
-   **Automotive (`automotive.py`) and Smart Home (`smart_home.py`):** (Future Concepts) Placeholder adapters for potential future integrations with IoT devices and vehicle control systems.

## 3. Interaction with Other Components

The adapters are the "hands" of AetherMind and are controlled by the `orchestrator`.

-   **`orchestrator/router.py`:** The router in the orchestrator is responsible for selecting the correct adapter based on the user's request and the `brain`'s generated Intent. For example, if the user is in a chat window, the `chat_ui.py` adapter will be chosen.
-   **`body/adapter_base.py`:** All adapters inherit from the `adapter_base` class, ensuring they conform to a standard interface that the `orchestrator` can reliably use.
-   **External World / User:** The output of an adapter is the final product that the user or an external system interacts with. The `chat_ui.py` adapter produces the text that appears in the user's chat window.

This directory is designed for extensibility. By simply adding a new file that inherits from `adapter_base.py`, AetherMind can be given a new voice, a new set of hands, or a new way of seeing the world.
