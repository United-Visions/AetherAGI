
"""
Path: orchestrator/agent_state_machine.py
Drop-in replacement for ActiveInferenceLoop.run_cycle
"""
import asyncio, json, time, redis.asyncio as redis
from enum import Enum
from loguru import logger
from typing import Optional

class State(Enum):
    WAITING   = "waiting"   # blocked on user
    PLANNING  = "planning"  # meta-controller choosing sub-system
    ACTING    = "acting"    # subsystem running
    LEARNING  = "learning"  # updating models
    PAUSED    = "paused"    # agent decided to sleep


class AgentStateMachine:
    """
    Persists after every transition to Redis JSON.
    Key = agent:{user_id}  value = serialized state
    """
    def __init__(self, redis_url: str, aether_loop, meta_ctrl):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.aether = aether_loop
        self.meta   = meta_ctrl

    async def run(self, user_id: str) -> None:
        """Main coroutine – runs forever for one user."""
        asyncio.create_task(self._listen_kill())
        while True:
            state = await self._load(user_id) or State.WAITING
            next_state = await self._tick(state, user_id)
            await self._save(user_id, next_state)
            await asyncio.sleep(0.1)   # event-loop yield

    async def _listen_kill(self):
        pub = self.redis.pubsub()
        await pub.subscribe("kill")
        async for msg in pub.listen():
            if msg["type"] == "message":
                logger.critical("KILL SWITCH ACTIVATED – shutting down")
                os._exit(42)

    # ------------------------------------------------------------------
    async def _tick(self, state: State, user_id: str) -> State:
        if state == State.WAITING:
            msg = await self._poll_user_message(user_id)  # non-blocking
            if msg:
                await self._append_stream(user_id, "user", msg)
                return State.PLANNING
            return State.WAITING

        if state == State.PLANNING:
            plan = await self.meta.decide_next_action(user_id)
            await self._append_stream(user_id, "assistant", json.dumps(plan))
            return State.ACTING

        if state == State.ACTING:
            action = await self._peek_last_agent_msg(user_id)
            result = await self._execute_action(action)
            await self._append_stream(user_id, "system", result)
            return State.LEARNING

        if state == State.LEARNING:
            # We assume aether loop has a way to pick up the last trace.
            # In the snippet it was just '...', so we simulate a call or skip if not implemented.
            # Since close_feedback_loop takes (message_id, score), and we don't have them here easily without more state,
            # we will assume this step is handled asynchronously or skipped for now in this state machine logic
            # until fully implemented.
            # await self.aether.close_feedback_loop(...)
            return State.WAITING

        return state

    # ----------------- small redis helpers ----------------------------
    async def _load(self, user_id: str) -> Optional[State]:
        raw = await self.redis.get(f"agent:{user_id}")
        return State(raw) if raw else None

    async def _save(self, user_id: str, state: State):
        await self.redis.set(f"agent:{user_id}", state.value, ex=86400)

    async def _append_stream(self, user_id: str, role: str, content: str):
        await self.redis.xadd(f"stream:{user_id}", {"role": role, "content": content})

    async def _peek_last_agent_msg(self, user_id: str) -> str:
        entries = await self.redis.xrevrange(f"stream:{user_id}", count=1)
        return entries[0][1]["content"] if entries else ""

    async def _execute_action(self, action_json: str) -> str:
        try:
            spec = json.loads(action_json)
            adapter = spec.get("adapter", "chat")
            return self.router.adapters[adapter].execute(spec["intent"])
        except Exception as e:
            return f"action crash: {e}"

    async def _poll_user_message(self, user_id: str) -> Optional[str]:
        # pop from redis stream – non-blocking
        msgs = await self.redis.xread({f"user:{user_id}": "$"}, count=1, block=100)
        if msgs:
            return msgs[0][1][0][1]["content"]
        return None
