"""
Path: curiosity/research_scheduler.py
Role: Manages an async job queue for research tasks using Redis.
"""
import redis.asyncio as redis
import json
from loguru import logger
from datetime import datetime

class ResearchScheduler:
    def __init__(self, redis_url: str):
        """
        Initializes the connection to the Redis server.
        Args:
            redis_url (str): The connection URL for the Redis server (e.g., "redis://localhost:6379").
        """
        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.queue_name = "research_job_queue"
            logger.success("Successfully connected to Redis for job scheduling.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {redis_url}: {e}")
            self.redis = None

    async def push(self, job: dict):
        """
        Pushes a new research job onto the Redis queue.
        The job's 'surprise' score is used as its priority.
        """
        if not self.redis:
            logger.error("Cannot push job: Redis connection not available.")
            return

        # Ensure job has the required fields
        job_schema = {
            "query": str, "surprise": float, "tools": list,
            "deadline": str, "user_id": str
        }
        for key, expected_type in job_schema.items():
            if key not in job or not isinstance(job[key], expected_type):
                logger.error(f"Job is missing or has malformed key: '{key}'. Aborting push.")
                return

        try:
            # The score for the sorted set is the surprise value. Higher score = higher priority.
            priority = job['surprise']
            job_json = json.dumps(job)
            
            await self.redis.zadd(self.queue_name, {job_json: priority})
            logger.info(f"Pushed research job with priority {priority:.4f}: '{job['query']}'")
            
        except Exception as e:
            logger.error(f"Failed to push job to Redis: {e}")

    async def pop(self) -> dict | None:
        """
        Pops the highest-priority research job from the queue.
        Uses ZREVRANGE with ZREM to atomically get and remove the top item.
        """
        if not self.redis:
            logger.error("Cannot pop job: Redis connection not available.")
            return None
        
        try:
            # Atomically get and remove the highest-scored item
            result = await self.redis.zpopmax(self.queue_name)
            if not result:
                return None
            
            # result is a list of tuples [(member, score)]
            job_json, score = result[0]
            job = json.loads(job_json)
            logger.info(f"Popped research job with priority {score:.4f}: '{job['query']}'")
            return job
            
        except Exception as e:
            logger.error(f"Failed to pop job from Redis: {e}")
            return None

    async def qsize(self) -> int:
        """Returns the current size of the job queue."""
        if not self.redis: return 0
        return await self.redis.zcard(self.queue_name)

