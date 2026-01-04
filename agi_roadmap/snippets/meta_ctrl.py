
"""
Path: orchestrator/meta_controller.py
Multi-armed bandit with UCB for now; replace by RL-on-latents later.
"""
import numpy as np
from typing import List
from mind.vector_store import AetherVectorStore
from brain.jepa_aligner import JEPAAligner

SUBSYSTEMS = ["chat", "plan", "practice", "imagine", "browse", "curiosity"]

class MetaController:
    def __init__(self, store: AetherVectorStore, jepa: JEPAAligner, budget: float):
        self.store = store
        self.jepa  = jepa
        self.budget = budget          # dollars left for user
        self.pulls = {s: 1 for s in SUBSYSTEMS}
        self.rewards = {s: 0.0 for s in SUBSYSTEMS}

    async def decide_next_action(self, user_id: str) -> dict:
        """
        Returns {"adapter": "plan", "intent": ..., "cost_usd": 0.012}
        """
        ctx_vec = await self._get_context_vector(user_id)
        ucbs = {}
        for s in SUBSYSTEMS:
            avg = self.rewards[s] / self.pulls[s]
            delta = np.sqrt(2 * np.log(sum(self.pulls.values())) / self.pulls[s])
            ucbs[s] = avg + delta

        choice = max(ucbs, key=ucbs.get)
        self.pulls[choice] += 1
        return {
            "adapter": choice,
            "intent": await self._craft_intent(choice, ctx_vec),
            "cost_usd": await self._estimate_cost(choice)
        }

    async def _get_context_vector(self, user_id: str) -> list:
        # last 3 messages from redis stream
        return [...]  # 1024-d

    async def _craft_intent(self, choice: str, ctx_vec: list) -> str:
        templates = {
            "plan": '{"adapter":"plan","intent":"generate plan for ..."}',
            "practice": '{"adapter":"practice","intent":"{\"language\":\"\",\"code\":\"...\"}"}',
            ...
        }
        return templates[choice]

    async def _estimate_cost(self, choice: str) -> float:
        cost_map = {"chat": 0.0005, "plan": 0.005, "practice": 0.002, "imagine": 0.001, "browse": 0.01, "curiosity": 0.02}
        return cost_map[choice]

    def update_reward(self, subsystem: str, reward: float):
        self.rewards[subsystem] += reward
 