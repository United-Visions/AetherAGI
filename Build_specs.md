# Build Specification: AetherMind Phase 1 (Linguistic Genesis)
**Version:** 1.0.0  
**Framework:** Developmental Continual Learning Architecture (DCLA)  
**Developer:** AetherGI

---

## 1. Project Vision
AetherMind is a **Digital Organism** designed to evolve from a foundational reasoning agent ("Baby") into a specialized expert. Phase 1 focuses on **Linguistic Genesis**: mastering logic, communication, and infinite episodic memory through a text-based interface.

---

## 2. Global Infrastructure & Services

| Component | Provider | Role |
| :--- | :--- | :--- |
| **The Brain (Inference)** | **RunPod** | GPU-accelerated Logic Engine (A100/L40). |
| **The Mind (Memory)** | **Pinecone** | Serverless Vector DB for Knowledge & Episodic Logs. |
| **The Orchestrator** | **Render** | FastAPI Backend (The Central Nervous System). |
| **The Interface** | **Vercel** | Next.js Frontend (The Chat Body). |
| **The Ingestor** | **FireCrawl** | URL-to-Markdown cleaning for data acquisition. |
| **User/Profile DB** | **Supabase** | Relational data for user IDs and "Body" settings. |
| **Version Control** | **GitHub** | Monorepo and CI/CD Pipelines. |

---

## 3. Component Specifications

### 3.1 The Brain (RunPod)
The cognitive core focusing on **how** to think, not **what** to think.
*   **Model:** Quantized Llama-3 or Mistral-7B (instruct-tuned for logic/reasoning).
*   **Safety Layer:** `safety_inhibitor.py` — A hard-coded classification layer running locally on the Pod that blocks any output violating "Do No Harm" before it leaves the Brain.
*   **Initialization:** Loaded with "Core Priors" (Logic trees and basic physics rules).
*   **Deployment:** Dockerized vLLM or FastAPI wrapper.

### 3.2 The Mind (Pinecone)
The infinite knowledge repository.
*   **Index Type:** Pinecone Serverless.
*   **Embedding Model (Dense):** `llama-text-embed-v2` (Hosted by Pinecone).
*   **Embedding Model (Sparse):** `pinecone-sparse-english-v0` (For keyword-exact recall).
*   **Memory Structure (Namespaces):**
    *   `core_k12`: Verified educational facts (Read-Only).
    *   `user_{id}_episodic`: Permanent log of every interaction for that specific user.
    *   `user_{id}_knowledge`: Custom banks (e.g., "Mechanic Mode") mounted by the user.

### 3.3 The Orchestrator (Render)
The logic hub that routes information.
*   **Framework:** Python FastAPI.
*   **Active Inference Loop:** Calculates the gap between user intent and internal model ("Surprise") and directs the Brain to close that gap.
*   **Context Assembly:** Queries Pinecone, fetches the top 10 relevant memories/facts, and builds the "Mega-Prompt" for the Brain.
*   **Hybrid Search:** Combines semantic (vibe) and sparse (keyword) results for 99.9% accuracy.

### 3.4 The Body: Interface (Vercel)
The physical/digital manifestation for Phase 1.
*   **Framework:** Next.js (React).
*   **Chat Body:** A specialized UI that handles text and voice input.
*   **Thought Bubble:** A side-panel that streams the Brain's "Internal Monologue" (e.g., *"Searching history for user's preference..."*).
*   **Body Adapters:** `adapter_base.py` — Ready-made code to plug the Brain into future bodies (IDE, Smart Home, Robot).

---

## 4. The Data Flow (The "Logic Loop")

1.  **Input:** User types "Remember my favorite color is blue" into the **Interface**.
2.  **Ingestion:** **Orchestrator** sends text to **Pinecone Inference API** to get vectors.
3.  **Context:** Orchestrator pulls relevant past "Blue" mentions from `user_episodic`.
4.  **Reasoning:** **Brain (RunPod)** receives the context. It logic-checks the input against its Safety Inhibitor.
5.  **Output:** Brain generates: "I have updated my world model. I now know you prefer blue."
6.  **Episodic Save:** The **Orchestrator** writes this interaction to the `user_episodic` namespace in Pinecone.
7.  **Display:** User sees the response + the "Thought Bubble" showing the update process.

---

## 5. Evolutionary Roadmap

### Phase 1: Linguistic Genesis (Current)
*   **Goal:** Perfect text logic and infinite memory.
*   **Body:** Chat/Text UI.

### Phase 2: Sensory Awakening
*   **Goal:** Multimodal understanding (Watching videos to learn).
*   **Addition:** JEPA Video/Image encoders added to the Brain.
*   **Body:** Camera/Visual integration.

### Phase 3: Phonetic Articulation
*   **Goal:** Human-like voice through phonetic learning.
*   **Addition:** The Brain learns to "sound out" words (vocal tract simulation) instead of using robotic TTS.
*   **Body:** Voice-first smart devices.

### Phase 4: Physical Embodiment
*   **Goal:** AGI in the physical world.
*   **Addition:** Motor Cortex and spatial navigation logic.
*   **Body:** **Humanoid Robot Body.** The same Brain from Phase 1 now controls limbs and sensors.

---

## 6. Development Steps (Immediate)

1.  **Repo Setup:** Initialize GitHub monorepo with the [Universal Structure].
2.  **Pinecone Init:** Create the `aethermind-genesis` index with `llama-text-embed-v2`.
3.  **FireCrawl Script:** Run `k12_ingestor.py` to populate the `core_k12` namespace with verified textbooks.
4.  **RunPod Deploy:** Spin up the logic-base model on an L40 GPU.
5.  **Orchestration:** Connect Render to RunPod and Pinecone via API keys.
6.  **Interface:** Deploy the Next.js frontend to Vercel and connect to Render.

---

## 7. Success Metrics
*   **No Forgetfulness:** 100% recall of user-stated facts after 1,000+ interactions.
*   **Causal Accuracy:** Ability to explain "Why" a physics problem results in a specific answer.
*   **Zero Hallucinations:** The Brain must state "I don't know" if the information is not in its **Mind** or **Core Priors**.

---
**AetherGI - Building Intelligence that Grows With You.**