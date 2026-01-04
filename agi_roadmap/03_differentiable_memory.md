
03_differentiable_memory.md


# 03 – Differentiable Memory (retrieval inside the compute graph)

Goal  
Make *“what to retrieve”* a **learned** decision by putting the top-k choice inside the **forward pass** so gradients flow back to the *query vector* and *top-k temperature*.

Why  
Without differentiable retrieval the agent cannot learn **what to remember** or **what to forget**—two core ingredients of human-level sample efficiency.

Files touched  
- mind/vector_store.py               (add soft top-k)  
- brain/logic_engine.py              (optional diff-ret path)  
- orchestrator/agent_state_machine.py (pass diff_ret flag)  

Feature flag  
SETTINGS.diff_retrieval = false