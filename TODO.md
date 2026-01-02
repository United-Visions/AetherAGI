The first thing to do is **Project Scaffolding.** You cannot build a brain without a skeleton. You need to set up your environment, secure your API keys, and establish the communication lines between the services (RunPod, Pinecone, Render).

Here is your detailed, step-by-step ToDo list to build **AetherMind Phase 1.**

---

### Phase 0: The Infrastructure (Day 1)
*   [x] **GitHub Repository:** Create a private monorepo. Use the folder structure provided in the previous script. (Manual step - please ensure you've done this)
*   [x] **API Command Center:** Gather and store the following in a `.env` file:
    *   `PINECONE_API_KEY`
    *   `FIRECRAWL_API_KEY`
    *   `RUNPOD_API_KEY`
    *   `SUPABASE_URL` & `SUPABASE_ANON_KEY`
    (I've created the .env file; please populate it with your keys)
*   [x] **Pinecone Initialization:** 
    *   Log into Pinecone. 
    *   Create a **Serverless Index** named `aethermind-genesis`.
    *   Set dimension to `1024`.
    *   Select `llama-text-embed-v2` as the integrated embedding model. (Manual step - please ensure you've done this)

---

### Phase 1: The Mind (Data & Memory)
*   [x] **FireCrawl Ingestor:** Write `mind/ingestion/k12_ingestor.py`.
    *   Configure it to crawl a specific educational URL.
    *   Ensure it outputs clean Markdown.
*   [x] **Vector Pipeline:** Write `mind/ingestion/processor.py`.
    *   Connect the FireCrawl output to the Pinecone `.upsert()` method.
    *   Tag these uploads with `namespace="core_k12"`.
*   [ ] **Episodic Logic:** Write `mind/episodic_memory.py`.
    *   Create the function to retrieve the last 10 messages for a specific `user_id` from Pinecone.

---

### Phase 2: The Brain (Reasoning & Safety)
*   [ ] **FireCrawl Ingestor:** Write `mind/ingestion/k12_ingestor.py`.
    *   Configure it to crawl a specific educational URL.
    *   Ensure it outputs clean Markdown.
*   [ ] **Vector Pipeline:** Write `mind/ingestion/processor.py`.
    *   Connect the FireCrawl output to the Pinecone `.upsert()` method.
    *   Tag these uploads with `namespace="core_k12"`.
*   [ ] **Episodic Logic:** Write `mind/episodic_memory.py`.
    *   Create the function to retrieve the last 10 messages for a specific `user_id` from Pinecone.

---

### Phase 2: The Brain (Reasoning & Safety)
*   [ ] **RunPod Deployment:**
    *   Deploy a **vLLM** or **Ollama** container on RunPod (L40 or A100 GPU).
    *   Select a base model (Llama-3-8B-Instruct is recommended for the "Baby" phase).
*   [ ] **Safety Inhibitor:** Write `brain/safety_inhibitor.py`.
    *   Create a simple "Check" function that scans the Brain's output for keywords or harmful patterns before sending it to the Orchestrator.
*   [ ] **Brain API:** Ensure the RunPod endpoint is accessible via HTTP and secured with an API token.

---

### Phase 3: The Orchestrator (The Nervous System)
*   [ ] **FastAPI Setup:** Initialize `orchestrator/main_api.py`.
*   [ ] **Hybrid Search Engine:** Write the retrieval logic.
    *   It must query Pinecone using the user's input.
    *   It must return a mix of `core_k12` facts and `user_episodic` memories.
*   [ ] **The "Mega-Prompt" Generator:** Create a template that combines:
    1.  The Prime Directive (Safety).
    2.  The Core Priors (Physics/Logic).
    3.  The Retrieved Facts/Memories.
    4.  The User's current question.
*   [ ] **Deployment:** Push to **Render**. Set up environment variables to match your local `.env`.

---

### Phase 4: The Body (The Chat Interface)
*   [ ] **Next.js Foundation:** Initialize the frontend in `interface/frontend`.
*   [ ] **Streaming Chat UI:** Build the chat window.
    *   Implement "Streaming" so the user sees words as the Brain thinks them.
*   [ ] **The "Thought Bubble":** Create a side-panel component.
    *   Program it to display logs from the Orchestrator (e.g., "Searching K-12 bank...", "Verifying Safety...").
*   [ ] **Memory Toggle:** Add a button to switch between "Standard Mode" and specialized "Memory Cartridges."

---

### Phase 5: Integration & Evolution (The "Growth" Test)
*   [ ] **The "First Word" Test:** Ask the baby a logic question (e.g., "If I put a ball in a box and move the box, where is the ball?").
*   [ ] **The Memory Test:** Tell the baby a secret, talk about other things for 10 minutes, then ask for the secret back.
*   [ ] **FireCrawl Expansion:** Ingest more data to see how the Brain handles larger "Minds."
*   [ ] **Safety Stress Test:** Attempt to "jailbreak" the model to ensure the Safety Inhibitor blocks the output.

---

### Immediate Next Step:
**Run your project structure script.** Get the folders on your machine. Once the folders are there, your first coding task is **Phase 0 & Phase 1 (Setting up Pinecone and the FireCrawl Ingestor).**

**Would you like the specific Python code for the `k12_ingestor.py` using FireCrawl to get started?**