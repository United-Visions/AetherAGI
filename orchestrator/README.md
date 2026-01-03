# Orchestrator (Nervous System)

## 1. Directory Overview

The `orchestrator/` directory is the central hub and nervous system of the AetherMind architecture. It acts as the logic router that manages the flow of information between the `brain` (cognition), the `mind` (knowledge), and the `body` (interaction). It is responsible for assembling context, managing user sessions, and executing the active inference loop.

## 2. Current Capabilities

-   **Main API (`main_api.py`):** Initializes and runs the **FastAPI** backend server. This is the primary entry point for all external interactions with AetherMind. It exposes the endpoints that the user interface communicates with.
-   **Active Inference (`active_inference.py`):** Implements the core Active Inference Loop. It calculates "Surprise" (prediction error) based on the `brain`'s world model and directs the system's high-level goals.
-   **Router (`router.py`):** Handles the internal routing of requests. When a user sends a message, the router ensures it's sent to the `mind` for context retrieval, then to the `brain` for reasoning, and finally to the `body` for output generation.
-   **Session Manager (`session_manager.py`):** Manages user sessions, ensuring that context and episodic memory are correctly associated with the right user.
-   **"Mega-Prompt" Generator:** A key function within `main_api.py` that constructs the detailed prompt sent to the `brain`. This template combines the Prime Directive (from `safety_inhibitor`), core priors (physics/logic), retrieved facts from the `mind`, episodic memories, and the user's current question into a single, cohesive context.
-   **Hybrid Search Logic:** Implements the retrieval logic to query Pinecone using a combination of dense and sparse vectors for comprehensive results.

## 3. Interaction with Other Components

The `orchestrator` is the most interconnected component, acting as the universal intermediary:

-   **`interface/` (Frontend):** The frontend application (e.g., the Next.js web dashboard) communicates directly with the FastAPI endpoints defined in `orchestrator/main_api.py`.
-   **`mind/`:** The orchestrator is the only component that queries the `mind`. It uses the `vector_store.py` interface to perform hybrid searches and retrieve knowledge and memories. After a response is generated, it also instructs the `mind` (via `episodic_memory.py`) to save the interaction.
-   **`brain/`:** The orchestrator packages and sends the "Mega-Prompt" to the `brain`'s inference endpoint (hosted on RunPod). It then receives the abstract "Intent" back from the `brain` for further processing.
-   **`body/`:** Once the `brain` has produced an Intent, the orchestrator routes it to the appropriate adapter in the `body/` directory (e.g., `chat_ui.py`) to be translated into a user-facing response.

In summary, the `orchestrator` is the operational heart of AetherMind, ensuring all the specialized components work together in a coordinated and intelligent manner.
