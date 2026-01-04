"""
Path: orchestrator/active_inference.py
Role: The Production Active Inference Loop.
"""

import numpy as np
import datetime
from brain.logic_engine import LogicEngine
from mind.episodic_memory import EpisodicMemory
from mind.vector_store import AetherVectorStore
from .router import Router
from heart.heart_orchestrator import Heart
from loguru import logger
from mind.promoter import Promoter
from heart.uncertainty_gate import UncertaintyGate
from config.settings import settings
import asyncio
import re

class ActiveInferenceLoop:
    def __init__(self, brain: LogicEngine, memory: EpisodicMemory, store: AetherVectorStore, router: Router, heart: Heart, surprise_detector=None):
        self.brain = brain
        self.memory = memory
        self.store = store
        self.router = router
        self.heart = heart
        self.surprise_detector = surprise_detector
        self.last_trace_data = {} # Simple cache for the trace
        self.promoter = Promoter(store, UncertaintyGate(self.heart.reward_model))

    async def run_cycle(self, user_id: str, user_input: str):
        """
        Production Loop with Full Heart Integration:
        Sense -> Reason -> Embellish -> Act -> Learn
        """
        logger.info(f"User {user_id} active inference started with Heart.")

        # 1. SENSE: Retrieve logical context
        k12_context, state_vec = self.store.query_context(user_input, namespace="core_universal")
        logger.debug(f"State vector shape: {len(state_vec)}")

        # Use EpisodicMemory wrapper to get timestamped context
        episodic_context = self.memory.get_recent_context(user_id, user_input)
        
        # 2. FEEL: Compute emotional and moral context from the Heart
        emotion_vector = self.heart.compute_emotion(user_input, user_id)
        predicted_flourishing = self.heart.predict_flourishing(state_vec)

        # Calculate surprise (Agent State)
        surprise_score = 0.0
        is_researching = False
        if self.surprise_detector:
             # Use the state vector for surprise calculation
             try:
                surprise_score = await self.surprise_detector.score(np.array(state_vec))
                is_researching = surprise_score > self.surprise_detector.novelty_threshold
             except Exception as e:
                 logger.warning(f"Surprise detection failed: {e}")

        agent_state = {
            "surprise_score": surprise_score,
            "is_researching": is_researching,
            "reason": "High novelty detected in context." if is_researching else "Routine interaction."
        }

        current_time_str = f"Current System Time: {datetime.datetime.now()}"
        combined_context = f"{current_time_str}\n" + "\n".join(k12_context + episodic_context)

        # 3. REASON: Brain processes input with all available context
        brain_response = await self.brain.generate_thought(
            user_input, 
            combined_context, 
            state_vec, 
            emotion_vector, # Pass the full vector
            predicted_flourishing
        )

        if "500" in brain_response: 
            return "The Brain is still waking up. Please wait 30 seconds and try again.", None, {}, {}

        # 4. EMBELLISH: Heart adapts the response based on emotion and morals
        embellished_response = self.heart.embellish_response(brain_response, emotion_vector, predicted_flourishing)

        # 5. ACT: Route the final, embellished response to the body
        final_output = self.router.forward_intent(embellished_response)

        # 6. PREPARE FOR LEARNING: Cache the data needed for the feedback loop
        self.last_trace_data[emotion_vector["message_id"]] = {
            "state_vector": state_vec,
            "action_text": final_output,
            "predicted_flourishing": predicted_flourishing
        }
        
        # 7. LEARN (Episodic): Save the interaction to memory
        self.memory.record_interaction(user_id, "user", user_input)
        self.memory.record_interaction(user_id, "assistant", final_output)
        logger.info(f"Successful interaction saved to user_{user_id}_episodic")

        logger.info(f"Cycle complete for User {user_id}")
        return final_output, emotion_vector["message_id"], emotion_vector, agent_state

    def close_feedback_loop(self, message_id: str, user_reaction_score: float):
        """
        Called by the API to close the learning loop with user feedback.
        """
        trace_data = self.last_trace_data.pop(message_id, None)
        if trace_data:
            self.heart.close_loop(trace_data, user_reaction_score)
            logger.success(f"Feedback loop closed for message {message_id}. Reward model updated.")

            if settings.promoter_gate:
                cleaned = re.sub(r"\S+@\S+", "", trace_data["action_text"])  # fast PII strip
                asyncio.create_task(
                    self.promoter.nugget_maybe_promote(
                        trace_data.get("user_id", "unknown"), # user_id is not in trace_data currently
                        cleaned,
                        surprise=abs(user_reaction_score - trace_data["predicted_flourishing"]),
                        flourish=user_reaction_score
                    )
                )
        else:
            logger.warning(f"Could not find trace data for message {message_id} to close feedback loop.")