
snippets/kill_switch.py


"""
Path: orchestrator/kill_switch.py
Big-red-button that **instantly** stops all agent loops.
"""
import os, redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class KillRequest(BaseModel):
    secret: str

KILL_SECRET = os.getenv("KILL_SECRET", "dev-secret-change-me")

app = FastAPI()

@app.post("/admin/kill")
async def kill_switch(req: KillRequest):
    if req.secret != KILL_SECRET:
        raise HTTPException(status_code=401, detail="wrong secret")
    r = redis.from_url(os.getenv("REDIS_URL"))
    await r.publish("kill", "1")          # pub-sub to all loops
    return {"status": "killed"}

# in agent_state_machine.py subscribe to channel and exit
#   async def _listen_kill(self):
#       pub = self.redis.pubsub()
#       await pub.subscribe("kill")
#       async for msg in pub.listen():
#           if msg["type"] == "message":
#               logger.critical("KILL SWITCH ACTIVATED – shutting down")
#               os._exit(42)