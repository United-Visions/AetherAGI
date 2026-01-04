
08_kill_switch.md


# 08 – Kill Switch (monitoring + big-red-button)

Goal  
Give humans **instant** visibility and **one-click shutdown** for any agent that starts **misbehaving** (budget spike, alignment drop, self-mod surge).

Why  
You are building AGI – **act like it**.

Files touched  
- monitoring/dashboard.py   (new)  
- orchestrator/kill_switch.py (new)  
- docker-compose.yml        (add prometheus + grafana)  

Feature flag  
SETTINGS.kill_switch = true   # leave ON always