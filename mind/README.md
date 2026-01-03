# Mind (Knowledge Repository)

## 1. Directory Overview

The `mind/` directory is the knowledge repository and long-term memory system for the AetherMind Digital Organism. It is the "expandable" part of the system, designed for potentially infinite knowledge storage and efficient retrieval. While the `brain` knows *how* to think, the `mind` is responsible for providing the information *what* to think about.

## 2. Current Capabilities

-   **Episodic Memory (`episodic_memory.py`):** Manages the "Digital Journal" of all interactions with a user. It stores conversational history, allowing AetherMind to recall past messages and maintain context over long periods. This module includes logic for "Dreaming/Consolidation," where long conversations are summarized into more compact "Knowledge Cartridges."
-   **Vector Store (`vector_store.py`):** Provides an interface to the **Pinecone Serverless** vector database. It handles the storage and retrieval of high-dimensional embeddings of all knowledge, from educational data to user-specific memories.
-   **Ingestion Sub-module (`ingestion/`):** Contains a suite of tools for populating the Mind's knowledge base. This includes crawlers for web content, processors for educational materials, and injectors for foundational axioms.

## 3. Interaction with Other Components

The `mind/` is the central library for AetherMind and is primarily controlled by the `orchestrator`.

-   **`orchestrator/`:** The `orchestrator` is the sole entry point to the `mind`. When a user provides input, the `orchestrator` queries the `mind`'s `vector_store` using hybrid search (combining dense and sparse vectors) to retrieve the most relevant information. This retrieved context, which can be a mix of core knowledge and episodic memories, is then passed to the `brain`.
-   **`brain/`:** The `brain` does not directly interact with the `mind`. It is a pure reasoning engine that operates on the data *provided by* the `mind` (via the `orchestrator`). This separation ensures that the reasoning process is not biased by the raw knowledge and allows the `brain` to remain a fixed, stable component.
-   **`body/`:** After the `brain` processes the information and generates a response, the `orchestrator` logs the entire interaction (user query and AetherMind's response) into the `mind`'s `episodic_memory` for future recall.

This architecture allows the `mind` to grow and learn continuously without altering the core logic of the `brain`. The `mind` provides the raw material of knowledge, which the `orchestrator` filters and organizes before the `brain` provides the spark of reason.
