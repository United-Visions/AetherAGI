
02_meta_controller.md


# 02 – Meta Controller (learned policy over subsystems)

Goal  
Replace hard-coded heuristics (`if "plan" in input`) with a **learned policy** π(subsystem | context) that maximises *expected progress per dollar*.

Why  
Humans don’t hard-code when to “think” vs “act”; they learn it.  
An AGI must learn *when* to plan, practice, imagine, browse, or ask.

Files touched  
- orchestrator/meta_controller.py   (new)  
- orchestrator/agent_state_machine.py (minor – call meta instead of if/else)  
- heart/reward_model.py             (minor – add subsystem reward)

Feature flag  
SETTINGS.meta_controller = false