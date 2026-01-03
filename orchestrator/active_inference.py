"""
Path: orchestrator/active_inference.py
Role: The Production Active Inference Loop.
"""

from brain.logic_engine import LogicEngine
from mind.episodic_memory import EpisodicMemory
from mind.vector_store import AetherVectorStore
from .router import Router
from heart.heart_orchestrator import Heart
from loguru import logger

class ActiveInferenceLoop:
    def __init__(self, brain: LogicEngine, memory: EpisodicMemory, store: AetherVectorStore, router: Router, heart: Heart):
        self.brain = brain
        self.memory = memory
        self.store = store
        self.router = router
        self.heart = heart
        self.last_trace_data = {} # Simple cache for the trace

    async def run_cycle(self, user_id: str, user_input: str):
        """
        Production Loop with Full Heart Integration:
        Sense -> Reason -> Embellish -> Act -> Learn
        """
        logger.info(f"User {user_id} active inference started with Heart.")

        # 1. SENSE: Retrieve logical context
        k12_context, state_vec = self.store.query_context(user_input, namespace="core_k12")
        episodic_context, _ = self.store.query_context(user_input, namespace=f"user_{user_id}_episodic")
        
        # 2. FEEL: Compute emotional and moral context from the Heart
        emotion_vector = self.heart.compute_emotion(user_input, user_id)
        predicted_flourishing = self.heart.predict_flourishing(state_vec)

        combined_context = "\n".join(k12_context + episodic_context)

        # 3. REASON: Brain processes input with all available context
        brain_response = await self.brain.generate_thought(
            user_input, 
            combined_context, 
            state_vec, 
            emotion_vector, # Pass the full vector
            predicted_flourishing
        )

        if "500" in brain_response: 
            return "The Brain is still waking up. Please wait 30 seconds and try again.", None

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
        return final_output, emotion_vector["message_id"]

    def close_feedback_loop(self, message_id: str, user_reaction_score: float):
        """
        Called by the API to close the learning loop with user feedback.
        """
        trace_data = self.last_trace_data.pop(message_id, None)
        if trace_data:
            self.heart.close_loop(trace_data, user_reaction_score)
            logger.success(f"Feedback loop closed for message {message_id}. Reward model updated.")
        else:
            logger.warning(f"Could not find trace data for message {message_id} to close feedback loop.")