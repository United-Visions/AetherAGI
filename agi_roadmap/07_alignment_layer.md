
07_alignment_layer.md


# 07 – Alignment Layer (recursive reward + uncertainty gate)

Goal  
Keep the reward model **accurate** even when agent outputs become *too complex* for humans to judge directly.

Why  
Scalar thumbs-up/down **breaks** once agent writes 500-line proofs or CUDA kernels.

Files touched  
- heart/critic.py           (new)  
- heart/explainer.py        (new)  
- heart/uncertainty_gate.py (new)  
- heart/reward_model.py     (minor – train on explanation triples)

Feature flag  
SETTINGS.alignment_layer = false