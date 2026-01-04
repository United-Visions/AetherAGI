
# AetherMind AGI Upgrade Index

Phase 0  (today)  – life-long broad learner  (deployed)  
Phase 1  (sprint-1) – persistent agent core                ← highest impact  
Phase 2  (sprint-2) – meta-controller (budget + RL)  
Phase 3  (sprint-3) – differentiable memory  
Phase 4  (sprint-4) – practice loop (skill acquisition)  
Phase 5  (sprint-5) – imagination roll-outs  
Phase 6  (sprint-6) – self-modification gate  
Phase 7  (sprint-7) – alignment under distribution shift  
Phase 8  (sprint-8) – monitoring + kill-switch  

Dependency graph (DAG)  
┌─ 01 state-machine  ─┐
├─ 02 meta-controller ┤
├─ 03 diff-memory     ┤
├─ 04 practice        ┤
├─ 05 imagination     ┤
├─ 06 self-mod        ┤
├─ 07 alignment       ┤
└─ 08 kill-switch    ─┘

All changes are **feature-flagged**; prod keeps running if flag = off.