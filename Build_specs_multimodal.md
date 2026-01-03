# AetherMind Phase 1.5: Sensory & Curiosity Roadmap

## 1. Overview

This document outlines the specifications and current implementation status for the **Sensory & Curiosity Upgrade**. It serves as a living roadmap to guide development from the current state (scaffolded with mock logic) to a fully operational, multimodal, and autonomous agent.

---

## 2. Implementation Status & Roadmap

This section details the status of each component and the specific tasks required to make it production-ready.

### 2.1 Perception System (`perception/`)

**Overall Status:** ✅ Scaffolding Complete, ⚠️ Logic is Partially Mocked

#### `perception/eye.py`
-   **Status:** ✅ **Implemented**
-   **Details:** The core logic for ingesting and processing images, video, audio, and PDFs is in place. It correctly routes different MIME types to the appropriate handlers.
-   **Next Steps:**
    -   **☐ VLM Integration:** The current VLM (BLIP) is functional but could be swapped for a more powerful model if needed.
    -   **☐ Error Handling:** Enhance error handling for corrupted files or unsupported codecs.

#### `perception/transcriber.py`
-   **Status:** ✅ **Implemented**
-   **Details:** The Whisper model integration is complete and functional for transcribing audio from various sources.
-   **Next Steps:**
    -   **☐ Model Optimization:** For production, evaluate if a larger Whisper model is needed for higher accuracy.

#### `perception/mcp_client.py`
-   **Status:** ⚠️ **Mock Implementation**
-   **Details:** The client simulates the MCP protocol. The `browse` tool has a functional connection to the FireCrawl API.
-   **Next Steps:**
    -   **☐ Implement `shell` tool:** Requires a secure sandbox environment for command execution.
    -   **☐ Implement `wolfram` tool:** Requires integrating the WolframAlpha API.
    -   **☐ Refine `browse` tool:** Improve the logic to use a search engine API (e.g., Google/Bing) to find relevant URLs before crawling them.

### 2.2 Curiosity System (`curiosity/`)

**Overall Status:** ✅ Scaffolding Complete, 🛑 Core Logic is Mocked

#### `curiosity/surprise_detector.py`
-   **Status:** 🛑 **Mock Implementation**
-   **Details:** This is the highest-priority component to update. It currently uses `MockJEPA` and `MockVectorStore` to simulate surprise calculations.
-   **Next Steps:**
    -   **☐ Integrate Real JEPA:** Replace `MockJEPA` with the actual `brain.jepa_aligner.JEPAAligner`. This involves connecting the main orchestration loop to feed state vectors into the detector.
    -   **☐ Integrate Real Vector Store:** Replace `MockVectorStore` with the production `mind.vector_store.VectorStore`. This requires initializing the store and performing a live query against the `autonomous_research` namespace.

#### `curiosity/research_scheduler.py`
-   **Status:** ✅ **Implemented**
-   **Details:** The Redis-based priority queue is fully implemented and ready for production.
-   **Next Steps:**
    -   **☐ Monitoring:** Add monitoring and logging to track queue size and job processing times.

#### `curiosity/solo_ingestor.py`
-   **Status:** ⚠️ **Partially Implemented**
-   **Details:** The worker loop, tool usage (`MCPClient`), and text processing are functional.
-   **Next Steps:**
    -   **☐ Enable Vector Upsert:** The final, critical step of storing the gathered knowledge is commented out. This needs to be uncommented and connected to the live `VectorStore` instance.
    -   **☐ Scalability:** For production, deploy the ingestor as a separate, scalable service (e.g., a dedicated Docker container).

### 2.3 Frontend Enhancements (`frontend_flask/`)

**Overall Status:** ✅ UI Complete, ⚠️ Backend Logic is Simulated

#### `templates/index.html` & `static/style.css`
-   **Status:** ✅ **Implemented**
-   **Details:** The chat interface now includes icons for camera, microphone, and file uploads. The UI is complete.
-   **Next Steps:** None.

#### `static/script.js`
-   **Status:** ⚠️ **Simulated Logic**
-   **Details:** The JavaScript correctly handles UI interactions (button clicks, file selection) and requests device permissions. However, the actual data handling is simulated.
-   **Next Steps:**
    -   **☐ Implement File Upload API:** Create a new endpoint in `frontend_flask/app.py` (e.g., `/v1/upload`) that can receive `FormData` file uploads.
    -   **☐ Implement Media Recording:** Use the `MediaRecorder` API to capture audio from the microphone and video from the camera, then send this data to the new upload endpoint.
    -   **☐ Replace `setTimeout`:** Connect the `handleFileUpload` function to the real upload endpoint instead of using a placeholder timer.

---

## 3. Original Specifications (Reference)

### 3.1 New Core Components
-   **Perception System (`perception/`)**: `eye.py`, `transcriber.py`, `mcp_client.py`.
-   **Curiosity System (`curiosity/`)**: `surprise_detector.py`, `research_scheduler.py`, `solo_ingestor.py`.

### 3.2 System Dependencies
-   `transformers`, `torch`, `sentencepiece`, `whisper-cpp-python`, `redis`, `av`, `pdfminer.six`, `pytesseract`, `beautifulsoup4`.

### 3.3 Data Flow: The Curiosity Loop
1.  **Input**: User provides multimodal input.
2.  **Ingestion**: `Eye` processes input into text and an embedding vector.
3.  **Surprise Detection**: `SurpriseDetector` scores the vector.
4.  **Question Generation**: Brain generates research questions.
5.  **Scheduling**: `ResearchScheduler` pushes jobs to the Redis queue.
6.  **Autonomous Research**: `SoloIngestor` uses `MCPClient` to find answers.
7.  **Knowledge Integration**: New information is stored in the `autonomous_research` vector namespace.
8.  **Enhanced Response**: The Orchestrator uses this new context to provide more informed answers.
