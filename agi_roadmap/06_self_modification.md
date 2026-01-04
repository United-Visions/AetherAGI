
06_self_modification.md


# 06 – Self-Modification Gate (safe code-gen + CI + hot-reload)

Goal  
Allow agent to **generate patches → run tests → merge → hot-reload** its *own* source while guaranteeing **rollback** on failure.

Why  
Recursive self-improvement is the **last mile** to AGI; it must be **sandboxed** and **reversible**.

Files touched  
- orchestrator/self_mod.py       (new)  
- orchestrator/router.py          (register "self_mod" adapter)  
- Dockerfile                      (add git + pytest in image)  

Feature flag  
SETTINGS.self_mod = false   # NEVER turn on without kill-switch & human approval