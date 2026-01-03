"""
Path: orchestrator/active_inference.py
Role: The Production Active Inference Loop.
"""

from brain.logic_engine import LogicEngine
from mind.episodic_memory import EpisodicMemory
from mind.vector_store import AetherVectorStore
from .router import Router
from loguru import logger

class ActiveInferenceLoop:
    def __init__(self, brain: LogicEngine, memory: EpisodicMemory, store: AetherVectorStore, router: Router):
        self.brain = brain
        self.memory = memory
        self.store = store
        self.router = router

    async def run_cycle(self, user_id: str, user_input: str):
        """
        Production Loop: Sense -> Reason -> Act -> Learn
        """
        logger.info(f"User {user_id} active inference started.")

        # 1. SENSE: Retrieve context + Current State Vector from the Mind
        k12_context, _ = self.store.query_context(user_input, namespace="core_k12")
        episodic_context, state_vec = self.store.query_context(user_input, namespace=f"user_{user_id}_episodic")

        combined_context = "\n".join(k12_context + episodic_context)

        # 2. REASON: Brain processes input vs context using JEPA and Priors
        brain_response = await self.brain.generate_thought(user_input, combined_context, state_vec)

        if "500" in brain_response: 
            return "The Brain is still waking up. Please wait 30 seconds and try again."

        # 3. ACT: Route the brain's intent to the appropriate body adapter
        final_output = self.router.forward_intent(brain_response)

        # 4. LEARN: The Memory Shield
        error_keywords = ["ERROR:", "SYSTEM ERROR", "UNREACHABLE", "500", "404"]
        is_valid_response = final_output and not any(k in final_output.upper() for k in error_keywords)

        if is_valid_response:
            self.memory.record_interaction(user_id, "user", user_input)
            self.memory.record_interaction(user_id, "assistant", final_output)
            logger.info(f"Successful interaction saved to user_{user_id}_episodic")
        else:
            logger.warning("Memory Shield active: Technical error detected, skipping storage to prevent mind corruption.")

        logger.info(f"Cycle complete for User {user_id}")
        return final_output