
01_state_machine.md

# 01 – Persistent Agent Core

Goal  
Replace the single-turn `ActiveInferenceLoop.run_cycle` with an **async state-machine** that:
- survives container restarts  
- interleaves *user* and *self-generated* messages in one stream  
- can pause/resume/migrate

Why  
Without persistent state the agent cannot pursue **multi-day goals** or *reflect* on its own past actions.

Files touched  
- orchestrator/active_inference.py   (major)  
- orchestrator/session_manager.py    (minor)  
- orchestrator/main_api.py           (minor – feature flag)

Feature flag  
SETTINGS.agent_state_machine = false  # default – old behaviour