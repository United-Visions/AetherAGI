# Copilot Instructions for AetherMind Phase 1: Linguistic Genesis

## 1. Project Vision and Architecture Overview

AetherMind is a **Digital Organism** designed for continuous learning and adaptation. Phase 1, "Linguistic Genesis," focuses on mastering logic, communication, and infinite episodic memory through a text-based interface. The core architecture separates **Reasoning (The Brain)** from **Knowledge (The Mind)** and **Interaction (The Body)**.

**References:**
- `WHITE_PAPER.MD`
- `readme.md`
- `Build_specs.md`
- `TODO.md`

## 2. Core Components and Responsibilities

### 2.1 The Brain (Cognitive Core - `brain/`)
- **Role:** The "fixed" part of the system responsible for *how* to think, not *what* to think. Contains logic, physics priors, active inference loop, and safety inhibitor.
- **Key Files:**
    - `brain/core_knowledge_priors.py`: Initializes the Brain with structural logic, mathematical proofs, programming syntax, and a causal world model (JEPA).
    - `brain/logic_engine.py`: Implements the active inference loop, minimizing "surprise" and driving goal-oriented responses.
    - `brain/safety_inhibitor.py`: A non-trainable classification layer that screens all intended outputs against the "Prime Directive" (Do No Harm). This is hard-wired and non-corruptible.
- **Technology:** Deployed on **RunPod** (GPU-accelerated, e.g., Llama-3-8B-Instruct via vLLM or Ollama).

### 2.2 The Mind (Knowledge Repository - `mind/`)
- **Role:** The "expandable" part of the system for infinite knowledge storage and retrieval.
- **Key Files:**
    - `mind/__init__.py`: Package initialization.
    - `mind/episodic_memory.py`: Manages the "Digital Journal" of user interactions, enabling semantic recency and retrieval of past messages. Includes "Dreaming/Consolidation" for summarizing long chats into "Knowledge Cartridges."
    - `mind/vector_store.py`: Interface to the Vector Database for storing high-dimensional embeddings of knowledge.
    - `mind/ingestion/web_crawler.py`: (Potentially to be implemented for general web crawling, currently FireCrawl is used externally).
    - `mind/ingestion/k12_ingestor.py`: (To be written) Configured to crawl educational URLs, output clean Markdown, and prepare data for Pinecone.
    - `mind/ingestion/processor.py`: (To be written) Connects FireCrawl output to Pinecone's `.upsert()` method, tagging uploads with `namespace="core_k12"`.
- **Technology:** **Pinecone Serverless** Vector DB (`aethermind-genesis` index, dimension `1024`, `llama-text-embed-v2` embedding model for dense, `pinecone-sparse-english-v0` for sparse).
- **Namespaces:** `core_k12`, `user_{id}_episodic`, `user_{id}_knowledge`.

### 2.3 The Orchestrator (The Nervous System - `orchestrator/`)
- **Role:** The logic hub that routes information between the Brain, Mind, and Body. Manages active inference, context assembly, and hybrid search.
- **Key Files:**
    - `orchestrator/__init__.py`: Package initialization.
    - `orchestrator/active_inference.py`: Implements the Active Inference Loop, calculating "Surprise" and directing the Brain.
    - `orchestrator/router.py`: Handles routing of information and requests.
    - `orchestrator/session_manager.py`: Manages user sessions and context.
    - `orchestrator/main_api.py`: (To be initialized) FastAPI setup for the backend. Includes retrieval logic for Pinecone (hybrid search), and the "Mega-Prompt" generator.
- **Technology:** Python **FastAPI**, deployed on **Render**.

### 2.4 The Body (Interface Layer - `body/` and `interface/`)
- **Role:** The "Enclosure" that allows the Brain to interact with the world. Phase 1 focuses on the Linguistic Interface (Chat UI).
- **Key Files:**
    - `body/__init__.py`: Package initialization.
    - `body/adapter_base.py`: Base class for body adapters.
    - `body/adapters/chat_ui.py`: (To be implemented) Translates Brain's abstract "Intents" into text tokens and voice synthesis for the chat interface.
    - `interface/thought_visualizer.py`: Creates a side-channel to visualize the Brain's reasoning (e.g., "Searching Mind for Newton's Second Law...").
    - `interface/web_dashboard/`: (Frontend)
        - `interface/web_dashboard/README.md`: Frontend documentation.
- **Technology:** **Next.js (React)** for frontend, deployed on **Vercel**.

## 3. Critical Developer Workflows

### 3.1 Project Setup and Environment Configuration (Phase 0)
1.  **GitHub Repository:** Ensure a private monorepo exists with the specified folder structure.
2.  **API Command Center:** Create a `.env` file at the root with the following keys:
    - `PINECONE_API_KEY`
    - `FIRECRAWL_API_KEY`
    - `RUNPOD_API_KEY`
    - `SUPABASE_URL`
    - `SUPABASE_ANON_KEY`
3.  **Pinecone Initialization:**
    - Log into Pinecone.
    - Create a **Serverless Index** named `aethermind-genesis`.
    - Set dimension to `1024`.
    - Select `llama-text-embed-v2` as the integrated embedding model.

### 3.2 Data Ingestion (Phase 1: Mind)
1.  **FireCrawl Ingestor (`mind/ingestion/k12_ingestor.py`):**
    - Implement a script to crawl a specific educational URL.
    - Ensure it outputs clean Markdown.
2.  **Vector Pipeline (`mind/ingestion/processor.py`):**
    - Connect the FireCrawl output to the Pinecone `.upsert()` method.
    - Tag these uploads with `namespace="core_k12"`.
3.  **Episodic Logic (`mind/episodic_memory.py`):**
    - Create a function to retrieve the last 10 messages for a specific `user_id` from Pinecone.

### 3.3 Brain Deployment and Safety (Phase 2: Brain)
1.  **RunPod Deployment:**
    - Deploy a **vLLM** or **Ollama** container on RunPod (L40 or A100 GPU).
    - Select a base model (Llama-3-8B-Instruct recommended).
2.  **Safety Inhibitor (`brain/safety_inhibitor.py`):**
    - Implement a "Check" function to scan Brain's output for keywords or harmful patterns before sending to Orchestrator.
3.  **Brain API:** Ensure the RunPod endpoint is accessible via HTTP and secured with an API token.

### 3.4 Orchestrator Backend (Phase 3: Orchestrator)
1.  **FastAPI Setup (`orchestrator/main_api.py`):**
    - Initialize the FastAPI application.
2.  **Hybrid Search Engine:**
    - Implement retrieval logic to query Pinecone using user input.
    - Return a mix of `core_k12` facts and `user_episodic` memories.
3.  **"Mega-Prompt" Generator:**
    - Create a template that combines: Prime Directive (Safety), Core Priors (Physics/Logic), Retrieved Facts/Memories, and the User's current question.
4.  **Deployment:** Push to **Render** and set up environment variables to match local `.env`.

### 3.5 Frontend Development (Phase 4: Body)
1.  **Next.js Foundation (`interface/frontend`):**
    - Initialize the Next.js frontend project.
2.  **Streaming Chat UI:**
    - Build the chat window with "Streaming" capability.
3.  **"Thought Bubble" Component:**
    - Create a side-panel component to display logs from the Orchestrator (e.g., "Searching K-12 bank...", "Verifying Safety...").
4.  **Memory Toggle:**
    - Add a button to switch between "Standard Mode" and specialized "Memory Cartridges."

## 4. Project-Specific Conventions and Patterns

-   **"Split-Brain" Architecture:** The clear separation of Brain (reasoning), Mind (knowledge), and Body (interface) is fundamental. Agents should respect these boundaries.
-   **Logic-First Initialization:** The Brain is *not* pretrained on raw web data. Its initial understanding comes from structural logic and causal world models.
-   **Hard-Wired Safety Inhibitor:** The `safety_inhibitor.py` is a critical, non-trainable layer. All Brain outputs *must* pass through this check.
-   **Infinite Episodic Memory:** Leverage `mind/episodic_memory.py` and Pinecone namespaces (`user_{id}_episodic`) for full conversational recall.
-   **Active Inference Loop:** The primary driver for the Brain's responses is minimizing "Surprise," not next-token prediction.
-   **Hybrid Search:** When querying Pinecone, use both dense (`llama-text-embed-v2`) and sparse (`pinecone-sparse-english-v0`) embeddings for comprehensive retrieval.
-   **"Mega-Prompt":** The Orchestrator constructs a detailed prompt for the Brain, incorporating safety, priors, and retrieved context.

## 5. Implementation Guidelines

-   **No Shortcuts or Brevity:** This is a complex system. Absolutely do not take any shortcuts or use brevity when implementing code. Every component and feature must be fully implemented according to the specifications in `WHITE_PAPER.MD`, `Build_specs.md`, and `TODO.md`.
-   **No Placeholders or Commented Out Sections:** Avoid any placeholder code, incomplete sections, or commented-out blocks that are not actively in use. All code provided must be fully functional and integrated into the architecture.
-   **Fully Implemented Code:** Provide complete, self-contained, and working code for every task. Do not provide partial implementations or outlines. The goal is a fully implemented Phase 1 AetherMind model.
-   **Detailed Logging:** Implement detailed logging for all critical flows, successful operations, errors, and intermediate states. This logging should be comprehensive enough to understand the system's behavior, debug issues, and trace data flow. Include relevant context in log messages (e.g., user IDs, component names, data values).

## 6. Integration Points and External Dependencies

-   **Pinecone:** Used extensively by `mind/` and `orchestrator/` for vector storage and retrieval. Requires `PINECONE_API_KEY`.
-   **RunPod:** Hosts the Brain's inference model. Requires `RUNPOD_API_KEY`.
-   **FireCrawl:** External service for web content ingestion. Requires `FIRECRAWL_API_KEY`.
-   **Render:** Hosts the FastAPI Orchestrator.
-   **Vercel:** Hosts the Next.js frontend.
-   **Supabase:** User and profile management. Requires `SUPABASE_URL` and `SUPABASE_ANON_KEY`.

## 6. Data Flow (The "Logic Loop")

Agents should understand the typical flow of information:

1.  **User Input (Interface):** User interaction through the `Body` (e.g., chat message).
2.  **Ingestion & Context (Orchestrator & Mind):** Orchestrator sends input for vectorization (Pinecone), retrieves relevant `core_k12` facts and `user_episodic` memories.
3.  **Reasoning & Safety (Brain):** Brain receives the assembled context. `safety_inhibitor.py` performs a critical check.
4.  **Output Generation (Brain):** Brain generates a logical, goal-oriented response.
5.  **Episodic Save (Orchestrator & Mind):** The interaction is logged to `user_{id}_episodic` in Pinecone.
6.  **Display (Interface):** The user sees the response, often accompanied by "Thought Bubble" visualizations.

## 7. Future Expansion (Beyond Phase 1)

While the immediate focus is Phase 1, the architecture is designed for:
-   **Phase 2: Sensory Awakening** (Visuals, JEPA for image/video).
-   **Phase 3: Phonetic Articulation** (Learning to speak phonetically).
-   **Phase 4: Physical Embodiment** (Humanoid Robotics).

Agents should be aware that their current work contributes to this larger vision.
## 8. Task Management and Progress Tracking

The `TODO.md` file in the root of the `aethermind_universal` directory contains a detailed, step-by-step ToDo list for building **AetherMind Phase 1**.

**Guidelines for using `TODO.md`:**
- **Follow Sequentially:** Work through the tasks in `TODO.md` in the order they are presented. Do not jump ahead unless explicitly instructed.
- **Check-off as Done:** Once a task is completed, update the `TODO.md` file by changing `[ ]` to `[x]` for the corresponding item. This provides clear progress tracking.
- **Refer to Details:** Each phase and task in `TODO.md` is accompanied by brief descriptions. Refer back to these and the `WHITE_PAPER.MD`, `readme.md`, and `Build_specs.md` for full context and specifications.
- **Phase Completion:** Ensure all tasks within a phase are completed and checked off before moving to the next phase.

**Example of checking off a task:**

**Before:**
```
### Phase 0: The Infrastructure (Day 1)
*   [ ] **GitHub Repository:** Create a private monorepo. Use the folder structure provided in the previous script.
```

**After completing the task:**
```
### Phase 0: The Infrastructure (Day 1)
*   [x] **GitHub Repository:** Create a private monorepo. Use the folder structure provided in the previous script.
```