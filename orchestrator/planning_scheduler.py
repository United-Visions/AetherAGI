"""
Path: orchestrator/planning_scheduler.py
Redis-backed long-horizon job queue for multi-day plans.
"""
import redis.asyncio as redis, json, uuid
from datetime import datetime, timedelta
from loguru import logger

class PlanningScheduler:
    def __init__(self, redis_url: str):
        self.r = redis.from_url(redis_url, decode_responses=True)
        self.q = "plan_queue"

    async def push_plan(self, user_id: str, plan: list, deadline_days: int = 7):
        job = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "plan": plan,
            "deadline": (datetime.utcnow() + timedelta(days=deadline_days)).isoformat(),
            "step_idx": 0
        }
        await self.r.zadd(self.q, {json.dumps(job): job["step_idx"]})
        logger.info(f"Pushed {len(plan)}-step plan for {user_id}")

    async def pop_next_step(self) -> dict | None:
        items = await self.r.zrange(self.q, 0, 0, withscores=True)
        if not items:
            return None
        job_json, _ = items[0]
        job = json.loads(job_json)
        await self.r.zrem(self.q, job_json)  # remove once popped
        return job

    async def reschedule_step(self, job: dict):
        job["step_idx"] += 1
        if job["step_idx"] < len(job["plan"]):
            await self.r.zadd(self.q, {json.dumps(job): job["step_idx"]})
