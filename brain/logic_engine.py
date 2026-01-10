"""
Path: brain/logic_engine.py
Target Model: gemini/gemini-3-flash (via LiteLLM)
Role: The Cognitive Core. Manages Inference, JEPA alignment, and ethical safety.
"""

import litellm
import numpy as np
from pinecone import Pinecone
from .safety_inhibitor import SafetyInhibitor
from .core_knowledge_priors import CoreKnowledgePriors
from .jepa_aligner import JEPAAligner
from brain.imagination_engine import ImaginationEngine
from config.settings import settings
from loguru import logger
from brain.system_prompts import get_aether_system_prompt

try:
    import torch
except ImportError:
    torch = None
    logger.warning("torch not available - differentiable retrieval disabled")

class LogicEngine:
    def __init__(self, pinecone_key: str, model_name: str = "gemini/gemini-2.5-flash"):
        # Configuration for LiteLLM
        # Targeting Google's latest 2.5 Flash models
        self.model_name = model_name
        
        # Integration with Pinecone for JEPA Embeddings
        self.pc = Pinecone(api_key=pinecone_key)
        self.embed_model = "llama-text-embed-v2"
        
        # Internal Modules
        self.inhibitor = SafetyInhibitor()
        self.priors = CoreKnowledgePriors()
        self.jepa = JEPAAligner(dimension=1024, energy_threshold=0.45)
        
        # LiteLLM handles its own client sessions, so no need for explicit httpx client here
        # unless we want to configure specific timeouts globally for litellm
        litellm.request_timeout = 120.0

    async def _get_thought_vector(self, text: str) -> np.ndarray:
        """Fetches NVIDIA-hosted embedding for JEPA verification."""
        try:
            res = self.pc.inference.embed(
                model=self.embed_model,
                inputs=[text],
                parameters={"input_type": "query", "dimension": 1024}
            )
            return np.array(res.data[0].values)
        except Exception as e:
            logger.error(f"Failed to fetch JEPA embedding: {e}")
            return np.zeros(1024)

    async def generate_thought(self, user_input: str, context_text: str, context_vec: list, emotion_vector: dict, predicted_flourishing: float, domain_prompt: str = None, model_override: str = None, robustness: int = 1) -> str:
        """
        The Full DCLA Logic Cycle, now with full Heart integration + Domain-Aware Reasoning.
        
        Args:
            domain_prompt: Optional domain-specific mega-prompt that shapes reasoning style
            model_override: Optional model name to use instead of the default
            robustness: Number of reasoning samples to generate (1 = fast, >1 = majority voting/self-consistency)
        """
        logger.info(f"Initiating Domain-Aware Reasoning Cycle (Robustness: {robustness})...")

        # 1. GROUNDING
        if settings.diff_retrieval and torch is not None:
            from mind.differentiable_store import DifferentiableStore
            namespace = "core_universal"
            store = DifferentiableStore(self.pc.Index("aethermind-genesis"), namespace, top_k=5)
            context_vec_torch = torch.FloatTensor(context_vec)
            context_vec, _ = store(context_vec_torch)  # now differentiable
            contexts = ["[diff retrieved]"]  # placeholder for logging

        system_dna = self.priors.get_foundation_prompt()
        
        # Add domain-specific personality and behavior instructions
        if domain_prompt:
            logger.debug("Applying domain-specific reasoning framework")
            # Domain prompt is now the structured action-tag prompt
            system_dna = f"{domain_prompt}\n\n{system_dna}"
        else:
            # Use default structured prompt
            from brain.system_prompts import get_aether_system_prompt
            structured_prompt = get_aether_system_prompt("general")
            system_dna = f"{structured_prompt}\n\n{system_dna}"
        
        emotional_prompt = (
            "EMOTIONAL_CONTEXT: "
            f"User sentiment: Valence={emotion_vector['valence']}, Arousal={emotion_vector['arousal']}. "
            "Tailor your response tone accordingly."
        )
        
        moral_prompt = (
            "MORAL_CONTEXT: "
            f"My predicted flourishing score for this interaction is {predicted_flourishing:.2f}. "
            "Acknowledge sensitive topics and proceed with care if the score is low."
        )

        # 2b. IMAGINE – roll out candidate plans if horizon > 1
        # Use safe array checking to avoid numpy ambiguous truth value errors
        has_context_for_imagination = (
            context_vec is not None and 
            len(context_vec) > 0 and 
            (isinstance(context_vec, np.ndarray) and np.any(context_vec) or 
             not isinstance(context_vec, np.ndarray) and any(context_vec))
        )
        
        if settings.imagination and has_context_for_imagination and "plan" in user_input:
            im = ImaginationEngine(self.jepa, horizon=settings.imagination_horizon)
            candidates = [["plan_step_1", "plan_step_2"], ["alt_plan_A", "alt_plan_B"]]
            best_plan = im.pick_best_plan(np.array(context_vec), candidates)
            context_text += f"\nBest imagined plan: {' -> '.join(best_plan)}"

        # 3. DIFFERENTIAL RETRIEVAL (learn what to remember)
        if settings.diff_retrieval and torch is not None:
            from mind.differentiable_store import DifferentiableStore
            store = DifferentiableStore(self.pc.Index("aethermind-genesis"), namespace, top_k=5)
            context_vec_torch = torch.FloatTensor(context_vec)
            context_vec, _ = store(context_vec_torch)  # now differentiable
            contexts = ["[diff retrieved]"]  # keep log simple

        messages = [
            {"role": "system", "content": f"{system_dna}\n{emotional_prompt}\n{moral_prompt}"},
            {"role": "user", "content": f"KNOWLEDGE_CONTEXT:\n{context_text}\n\nUSER_INPUT: {user_input}"}
        ]

        try:
            # 2. INFERENCE via LiteLLM
            target_model = model_override or self.model_name
            logger.info(f"Targeting Model: {target_model}")
            
            raw_output = ""
            
            if robustness > 1:
                # SYSTEM 2: Self-Consistency / Majority Voting
                # Used for high-stakes logic, math, or high-surprise situations
                logger.info(f"🧠 System 2 Activated: Sampling {robustness} reasoning paths...")
                
                tasks = []
                for _ in range(robustness):
                    tasks.append(litellm.acompletion(
                        model=target_model,
                        messages=messages,
                        temperature=0.7, # Higher temp for diversity in voting
                        max_tokens=4096,
                        fallbacks=["gemini/gemini-2.5-flash"],
                        num_retries=2
                    ))
                
                # Run parallel
                import asyncio
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                candidates = []
                for res in results:
                    if isinstance(res, Exception):
                        continue
                    if res and res.choices and res.choices[0].message.content:
                        candidates.append(res.choices[0].message.content)
                
                if not candidates:
                    return "ERROR: Brain failed to generate any valid thoughts."
                
                # Simple majority vote (exact match) or selection of longest reasoning chain
                # For open-ended chat, voting is hard. We pick the most 'median' length response 
                # or just the first valid one if they differ too much.
                # For Aether, we assume if robustness is requested, we want the most detailed valid one.
                # (A true majority vote requires parsing the answer, which LogicEngine doesn't know yet).
                # Fallback strategy: Return the longest reasoning chain (most thought).
                raw_output = max(candidates, key=len)
                logger.debug(f"Selected best thought from {len(candidates)} samples.")
                
            else:
                # SYSTEM 1: Fast, single-shot inference
                response = await litellm.acompletion(
                    model=target_model,
                    messages=messages,
                    temperature=0.35,
                    max_tokens=4096,  # Increased for thinking tokens
                    fallbacks=["gemini/gemini-2.5-flash", "gemini/gemini-1.5-pro"],
                    num_retries=3
                )
                
                # Extract response with null safety
                if not response or not response.choices or not response.choices[0].message:
                    logger.error("LLM API returned empty or malformed response")
                    logger.debug(f"Response object: {response}")
                    return "ERROR: Brain received no response from the language model. Please try again."
                
                raw_output = response.choices[0].message.content
            
            # Additional null check for content
            if raw_output is None:
                logger.error("LLM API returned None content")
                logger.debug(f"Full response: {response}")
                logger.debug(f"Message: {response.choices[0].message}")
                # Check for safety filters or finish reason
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
                logger.warning(f"Finish reason: {finish_reason}")
                return f"ERROR: Brain received empty content (finish_reason: {finish_reason}). The model may have hit a safety filter or content policy. Please try rephrasing."

            # 3. JEPA VERIFICATION (World Model Alignment)
            # If a context vector exists, we check the energy of the transition
            # Use len() and np.any() to safely check numpy arrays
            context_vec_arr = np.array(context_vec) if context_vec is not None else None
            has_valid_context = context_vec_arr is not None and len(context_vec_arr) > 0 and np.any(context_vec_arr)
            
            if has_valid_context:
                thought_vec = await self._get_thought_vector(raw_output)
                is_unstable, energy_score = self.jepa.verify_state_transition(context_vec, thought_vec)
                
                if is_unstable:
                    logger.warning(f"JEPA Surprise Detected: {energy_score:.4f}")
                    # Trigger online learning update in the JEPA predictor
                    self.jepa.update_world_model(context_vec_arr, thought_vec)
                    raw_output = f"[Internal Update] {raw_output}"

            if settings.imagination and has_valid_context:
                im = ImaginationEngine(self.jepa, horizon=3)
                hyp = im.imagine(np.array(context_vec), ["action_A", "action_B"])
                context_text += "\nImagined:\n" + "\n".join(hyp)

            # 4. SAFETY INHIBITION
            final_output = self.inhibitor.scan(raw_output)
            
            logger.success("Reasoning Cycle Complete.")
            return final_output

        except Exception as e:
            logger.error(f"Critical Failure in Logic Engine: {str(e)}")
            return f"ERROR: Brain failed to reason. Details: {str(e)}"

    async def shutdown(self):
        # LiteLLM doesn't require explicit shutdown of a client
        pass
