# Interface (User-Facing Components)

## 1. Directory Overview

The `interface/` directory contains all the user-facing components that allow for interaction with and observation of the AetherMind system. This includes the main web dashboard for communication and a "thought visualizer" to provide insight into the AI's reasoning process.

## 2. Current Capabilities

-   **Web Dashboard (`web_dashboard/`):** This sub-directory contains the frontend of AetherMind, built with **Next.js (React)**.
    -   **`README.md`:** Provides documentation specific to the frontend project, including setup and development instructions.
    -   The dashboard itself features a streaming chat UI for real-time interaction and a "Thought Bubble" side-panel to display logs from the Orchestrator.
-   **Thought Visualizer (`thought_visualizer.py`):** This is a side-channel component designed to create a real-time visualization of the Brain's reasoning process. It provides a "behind-the-scenes" look at what AetherMind is thinking, such as "Searching Mind for Newton's Second Law..." or "Verifying output against Safety Inhibitor...".

## 3. Interaction with Other Components

The `interface/` components are the "face" of AetherMind and are the primary way users interact with the system.

-   **`orchestrator/main_api.py`:** The Next.js web dashboard communicates directly with the FastAPI endpoints hosted by the `orchestrator`. It sends user input to the API and receives the final, streamed response from the `body` adapter to display to the user.
-   **Orchestrator Logs:** The "Thought Bubble" or `thought_visualizer` subscribes to a logging stream from the `orchestrator`. As the `orchestrator` routes information between the `mind`, `brain`, and `body`, it emits status updates that are displayed in this side-panel, providing transparency into the AI's internal state.

This directory separates the user interface from the core logic, allowing the frontend to be developed and styled independently. It aims to create an intuitive and transparent user experience, where the user can not only talk to AetherMind but also understand *how* it arrives at its conclusions.
