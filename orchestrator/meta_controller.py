"""
Path: orchestrator/meta_controller.py
UCB bandit that picks the next subsystem to maximise reward-per-dollar.
"""
import os
import math
from typing import Dict
from mind.vector_store import AetherVectorStore
from brain.jepa_aligner import JEPAAligner
from loguru import logger

SUBSYSTEMS = ["chat", "plan", "practice", "imagine", "browse", "curiosity"]

# cost in USD per call (rough)
COST_MAP = {
    "chat": 0.0005,
    "plan": 0.005,
    "practice": 0.002,
    "imagine": 0.001,
    "browse": 0.01,
    "curiosity": 0.02,
}

class MetaController:
    def __init__(self, store: AetherVectorStore, jepa: JEPAAligner):
        self.store = store
        self.jepa  = jepa
        # running stats
        self.pulls: Dict[str, int] = {s: 1 for s in SUBSYSTEMS}
        self.rewards: Dict[str, float] = {s: 0.0 for s in SUBSYSTEMS}

    async def decide_next_action(self, user_id: str) -> dict:
        """Returns {'adapter': <name>, 'intent': <json>, 'cost_usd': <float>}"""
        ctx_vec = await self._get_user_context(user_id)
        total   = sum(self.pulls.values())

        best_s, best_ucb = None, -math.inf
        for s in SUBSYSTEMS:
            avg = self.rewards[s] / self.pulls[s]
            ucb = avg + math.sqrt(2 * math.log(total) / self.pulls[s])
            if ucb > best_ucb:
                best_ucb = ucb
                best_s   = s

        self.pulls[best_s] += 1
        intent = self._template_intent(best_s, ctx_vec)
        logger.info(f"MetaController chose {best_s} (ucb={best_ucb:.3f})")
        return {"adapter": best_s, "intent": intent, "cost_usd": COST_MAP[best_s]}

    # ----------- helpers -----------
    async def _get_user_context(self, user_id: str) -> list:
        # last 3 messages from user episodic namespace
        ns = f"user_{user_id}_episodic"
        texts, _ = self.store.query_context("__last_turn__", namespace=ns, top_k=3)
        return texts  # list[str]

    def _template_intent(self, adapter: str, ctx: list) -> str:
        templates = {
            "chat": '{"role":"user","content":"__last_user_msg__"}',
            "plan": '{"adapter":"plan","intent":"generate high-level steps"}',
            "practice": '{"adapter":"practice","language":"","code":""}',
            "imagine": '{"adapter":"imagine","actions":["action_A","action_B"]}',
            "browse": '{"adapter":"browse","url":"<searchTerm>"}',
            "curiosity": '{"adapter":"curiosity","query":"surprise>0.5"}',
        }
        return templates.get(adapter, "{}")

    def update_reward(self, adapter: str, reward: float):
        """reward in [-1, 1]"""
        self.rewards[adapter] += reward
        logger.debug(f"MetaController reward {adapter} += {reward}")
