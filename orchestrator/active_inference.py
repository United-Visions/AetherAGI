"""
Path: orchestrator/active_inference.py
Role: The Production Active Inference Loop with Domain-Aware Reasoning.
"""

import numpy as np
import datetime
from brain.logic_engine import LogicEngine
from mind.episodic_memory import EpisodicMemory
from mind.vector_store import AetherVectorStore
from .router import Router
from .session_manager import SessionManager
from heart.heart_orchestrator import Heart
from loguru import logger
from mind.promoter import Promoter
from heart.uncertainty_gate import UncertaintyGate
from config.settings import settings
import asyncio
import re

class ActiveInferenceLoop:
    def __init__(self, brain: LogicEngine, memory: EpisodicMemory, store: AetherVectorStore, router: Router, heart: Heart, surprise_detector=None, session_manager=None):
        self.brain = brain
        self.memory = memory
        self.store = store
        self.router = router
        self.heart = heart
        self.surprise_detector = surprise_detector
        self.session_manager = session_manager or SessionManager()
        self.last_trace_data = {} # Simple cache for the trace
        self.promoter = Promoter(store, UncertaintyGate(self.heart.reward_model))

    async def run_cycle(self, user_id: str, user_input: str, namespace: str = "universal"):
        """
        Production Loop with Full Heart Integration + Domain-Aware Reasoning:
        Sense -> Reason (Domain-Focused) -> Embellish -> Act -> Learn
        """
        # Get user's domain profile
        user_profile = self.session_manager.get_user_profile(user_id)
        domain_profile = user_profile["domain_profile"]
        domain = user_profile["domain"]
        
        logger.info(f"User {user_id} active inference started ({domain_profile.display_name})")

        # 1. SENSE: Retrieve context with domain-weighted namespaces
        namespace_weights = domain_profile.namespace_weights
        k12_context, state_vec = await self._domain_aware_context_retrieval(
            user_input, 
            user_id,
            namespace_weights
        )
        logger.debug(f"State vector shape: {len(state_vec)}, Domain: {domain}")

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
            "reason": "High novelty detected in context." if is_researching else "Routine interaction.",
            "domain": domain,
            "domain_display_name": domain_profile.display_name
        }

        current_time_str = f"Current System Time: {datetime.datetime.now()}"
        combined_context = f"{current_time_str}\n" + "\n".join(k12_context + episodic_context)

        # 3. BUILD DOMAIN-SPECIFIC MEGA-PROMPT
        domain_mega_prompt = self.session_manager.get_mega_prompt_prefix(user_id)
        
        # 4. REASON: Brain processes input with domain-aware context and personality
        brain_response = await self.brain.generate_thought(
            user_input, 
            combined_context, 
            state_vec, 
            emotion_vector, # Pass the full vector
            predicted_flourishing,
            domain_prompt=domain_mega_prompt  # NEW: Domain-specific instruction
        )

        if "500" in brain_response: 
            return "The Brain is still waking up. Please wait 30 seconds and try again.", None, {}, {}

        # 5. EMBELLISH: Heart adapts the response based on emotion and morals
        embellished_response = self.heart.embellish_response(brain_response, emotion_vector, predicted_flourishing)

        # 6. ACT: Route the final, embellished response to the body
        final_output = self.router.forward_intent(embellished_response)

        # 7. PREPARE FOR LEARNING: Cache the data needed for the feedback loop
        self.last_trace_data[emotion_vector["message_id"]] = {
            "state_vector": state_vec,
            "action_text": final_output,
            "predicted_flourishing": predicted_flourishing,
            "user_id": user_id  # Store for promoter
        }
        
        # 8. LEARN (Episodic): Save the interaction to memory
        self.memory.record_interaction(user_id, "user", user_input)
        self.memory.record_interaction(user_id, "assistant", final_output)
        logger.info(f"Successful interaction saved to user_{user_id}_episodic")
        
        # 9. UPDATE USER LEARNING CONTEXT
        self.session_manager.update_learning_context(user_id, {
            "topic": user_input[:100],  # First 100 chars as topic
            "domain_relevant": True,
            "tools_used": [],  # TODO: Track actual tools used
            "cross_domain": False  # TODO: Detect cross-domain queries
        })

        logger.info(f"Cycle complete for User {user_id} ({domain_profile.display_name})")
        return final_output, emotion_vector["message_id"], emotion_vector, agent_state
    
    async def _domain_aware_context_retrieval(self, user_input: str, user_id: str, namespace_weights: dict):
        """
        Retrieves context from multiple namespaces with domain-specific weighting.
        
        Args:
            user_input: The user's query
            user_id: User identifier for personalized namespaces
            namespace_weights: Dict of namespace -> weight (e.g., {"core_universal": 0.2, "domain_code": 0.6})
        
        Returns:
            (combined_contexts: List[str], state_vector: List[float])
        """
        all_contexts = []
        weighted_vectors = []
        
        for namespace, weight in namespace_weights.items():
            if weight == 0:
                continue
                
            try:
                # Handle user-specific namespaces
                if namespace.startswith("user_"):
                    actual_namespace = f"user_{user_id}_{namespace.split('_')[-1]}"
                else:
                    actual_namespace = namespace
                
                # Query this namespace
                contexts, state_vec = self.store.query_context(
                    user_input, 
                    namespace=actual_namespace,
                    top_k=max(1, int(5 * weight))  # More results for higher-weighted namespaces
                )
                
                # Weight the contexts
                all_contexts.extend([f"[{namespace}] {ctx}" for ctx in contexts])
                
                # Weight the state vector
                if state_vec and len(state_vec) > 0:
                    weighted_vectors.append((np.array(state_vec), weight))
                    
            except Exception as e:
                logger.warning(f"Failed to retrieve from namespace {namespace}: {e}")
        
        # Combine weighted vectors
        if weighted_vectors:
            final_vec = sum(vec * w for vec, w in weighted_vectors) / sum(w for _, w in weighted_vectors)
            final_vec = final_vec.tolist()
        else:
            # Fallback to core_universal if all else fails
            logger.warning("No weighted vectors found, falling back to core_universal")
            all_contexts, final_vec = self.store.query_context(user_input, namespace="core_universal")
        
        logger.debug(f"Retrieved {len(all_contexts)} contexts from {len(namespace_weights)} namespaces")
        return all_contexts, final_vec

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
                        trace_data.get("user_id", "unknown"),
                        cleaned,
                        surprise=abs(user_reaction_score - trace_data["predicted_flourishing"]),
                        flourish=user_reaction_score
                    )
                )
        else:
            logger.warning(f"Could not find trace data for message {message_id} to close feedback loop.")