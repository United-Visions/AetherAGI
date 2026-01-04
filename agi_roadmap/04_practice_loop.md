


04_practice_loop.md


# 04 – Practice Loop (skill acquisition by doing)

Goal  
Let the agent *write code / run shell / control browser* and observe **stdout / stderr / screenshots** so it can **improve procedures** through trial-and-error.

Why  
Reading ten  books ≠ being able to debug.  AGI must *practice*.

Files touched  
- body/adapters/practice_adapter.py  (new)  
- orchestrator/router.py              (register adapter)  
- orchestrator/agent_state_machine.py (call practice)  

Feature flag  
SETTINGS.practice_adapter = false