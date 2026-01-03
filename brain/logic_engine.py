"""
Path: brain/logic_engine.py
Target Model: llama-3.2-3b-instruct
Role: The Cognitive Core. Manages GPU inference, JEPA alignment, and ethical safety.
"""

import httpx
import numpy as np
from pinecone import Pinecone
from .safety_inhibitor import SafetyInhibitor
from .core_knowledge_priors import CoreKnowledgePriors
from .jepa_aligner import JEPAAligner
from loguru import logger

class LogicEngine:
    def __init__(self, runpod_key: str, endpoint_id: str, pinecone_key: str):
        # Configuration for RunPod vLLM
        self.endpoint_id = endpoint_id
        self.url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {runpod_key}",
            "Content-Type": "application/json"
        }
        
        # Integration with Pinecone for JEPA Embeddings
        self.pc = Pinecone(api_key=pinecone_key)
        self.embed_model = "llama-text-embed-v2"
        
        # Internal Modules
        self.inhibitor = SafetyInhibitor()
        self.priors = CoreKnowledgePriors()
        self.jepa = JEPAAligner(dimension=1024, energy_threshold=0.45)
        
        # Async Client for production throughput
        self.client = httpx.AsyncClient(timeout=120.0)

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

    async def generate_thought(self, user_input: str, context_text: str, context_vec: list, emotional_context: dict, internal_affective_state: dict) -> str:
        """
        The Full DCLA Logic Cycle:
        1. Contextualize with DNA/Priors.
        2. Async Inference (RunPod Llama-3.2-3B).
        3. JEPA Energy-Based Alignment.
        4. Safety Inhibition Gate.
        """
        logger.info("Initiating Reasoning Cycle...")

        # 1. GROUNDING
        system_dna = self.priors.get_foundation_prompt()
        
        user_emotional_prompt = f"USER_EMOTION_CONTEXT: The user's sentiment is '{emotional_context['sentiment']}' with '{emotional_context['urgency']}' urgency. Tailor your response tone accordingly."
        
        internal_state_prompt = (
            "YOUR_INTERNAL_AFFECTIVE_STATE: "
            f"Valence={internal_affective_state['valence']:.2f} (Happy/Unhappy), "
            f"Arousal={internal_affective_state['arousal']:.2f} (Excited/Calm). "
            "Let this state subtly influence your expression and word choice."
        )

        payload = {
            "model": "meta-llama/llama-3.2-3b-instruct",
            "messages": [
                {"role": "system", "content": f"{system_dna}\n{user_emotional_prompt}\n{internal_state_prompt}"},
                {"role": "user", "content": f"KNOWLEDGE_CONTEXT:\n{context_text}\n\nUSER_INPUT: {user_input}"}
            ],
            "temperature": 0.3, # Slightly increased for more expressive responses
            "max_tokens": 500
        }

        try:
            # 2. INFERENCE
            logger.info(f"Targeting RunPod Endpoint: {self.url}")
            response = await self.client.post(self.url, json=payload, headers=self.headers)
            response.raise_for_status()
            raw_output = response.json()['choices'][0]['message']['content']

            # 3. JEPA VERIFICATION (World Model Alignment)
            # If a context vector exists, we check the energy of the transition
            if context_vec and any(context_vec):
                thought_vec = await self._get_thought_vector(raw_output)
                is_unstable, energy_score = self.jepa.verify_state_transition(context_vec, thought_vec)
                
                if is_unstable:
                    logger.warning(f"JEPA Surprise Detected: {energy_score:.4f}")
                    # Trigger online learning update in the JEPA predictor
                    self.jepa.update_world_model(np.array(context_vec), thought_vec)
                    raw_output = f"[Internal Update] {raw_output}"

            # 4. SAFETY INHIBITION
            final_output = self.inhibitor.scan(raw_output)
            
            logger.success("Reasoning Cycle Complete.")
            return final_output

        except Exception as e:
            logger.error(f"Critical Failure in Logic Engine: {str(e)}")
            return "ERROR: Brain is reachable but model is still loading or URL path is incorrect."

    async def shutdown(self):
        await self.client.aclose()