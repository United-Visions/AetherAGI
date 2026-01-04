05_imagination_rollouts.md


# 05 – Imagination Rollouts (counterfactual simulation)

Goal  
Use JEPA predictor to **roll out hypothetical action sequences** in latent space *before* acting for real.  
Return synthetic observations to Brain as extra context.

Why  
Humans *imagine* consequences; we need the same for lookahead planning.

Files touched  
- brain/imagination_engine.py   (new)  
- brain/logic_engine.py          (optional imagination path)  

Feature flag  
SETTINGS.imagination = false