# Brain (Cognitive Core)

## 1. Directory Overview

The `brain/` directory serves as the cognitive core of the AetherMind Digital Organism. This is the "fixed" or non-trainable part of the system, responsible for *how* to think, not *what* to think. It contains the fundamental logic, reasoning capabilities, and safety mechanisms that govern AetherMind's behavior.

## 2. Current Capabilities

-   **Core Knowledge Priors (`core_knowledge_priors.py`):** Initializes the Brain with a foundational understanding of logic, mathematics, and basic physics. This serves as the bedrock of its reasoning, ensuring it has a consistent and rational framework from which to operate.
-   **Logic Engine (`logic_engine.py`):** Implements an active inference loop. This is the primary driver of the Brain's cognitive processes, constantly working to minimize "surprise" (prediction error) and generate goal-oriented responses. It allows AetherMind to reason, plan, and make decisions.
-   **Safety Inhibitor (`safety_inhibitor.py`):** A critical, non-trainable classification layer that screens all intended outputs against the "Prime Directive" (Do No Harm). This hard-wired component ensures that all of AetherMind's actions are safe and ethical.
-   **JEPA Aligner (`jepa_aligner.py`):** (In development) A component designed to build a causal world model based on the Joint-Embedding Predictive Architecture, allowing for more abstract and predictive reasoning.

## 3. Interaction with Other Components

The `brain/` directory is the central processing unit and does not operate in isolation. It interacts closely with other key modules:

-   **`orchestrator/`:** The `orchestrator` acts as the nervous system, feeding the `brain` a carefully constructed "Mega-Prompt." This prompt contains context from the `mind` (retrieved memories and knowledge) and the user's current query. The `brain` processes this information and returns its logical output (an "Intent") back to the `orchestrator`.
-   **`mind/`:** The `brain` does not store long-term knowledge. Instead, it relies on the `orchestrator` to query the `mind`'s vector store for relevant information. The `brain`'s role is to reason over the data provided by the `mind`, not to memorize it.
-   **`body/`:** The `brain` generates abstract "Intents," which are then passed to the `orchestrator`. The `orchestrator` routes these Intents to the appropriate adapter in the `body/` directory (e.g., `chat_ui.py`), which translates the abstract thought into a concrete action, like generating a text response.

The `brain` is designed to be a pure reasoning engine, decoupled from knowledge storage and physical interaction. This "Split-Brain" architecture allows for modular development and ensures that the core logic remains stable and secure.
