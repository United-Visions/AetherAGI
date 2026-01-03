"""
Path: curiosity/surprise_detector.py
Role: Decides what surprises are worth researching using a JEPA energy + novelty hybrid.
"""
import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from cachetools import TTLCache

from mind.vector_store import AetherVectorStore
from brain.jepa_aligner import JEPAAligner

class SurpriseDetector:
    def __init__(self, jepa: JEPAAligner, store: AetherVectorStore, novelty_threshold=0.5):
        self.jepa = jepa # Now using the real, injected JEPA Aligner
        self.store = store
        self.novelty_threshold = novelty_threshold
        self.cache = TTLCache(maxsize=10_000, ttl=3600 * 24) # 24-hour cache
        self.last_vec = np.random.rand(1024) # Initialize with a random vector

    async def score(self, new_vec: np.ndarray) -> float:
        """
        Scores a new vector for surprise based on JEPA energy and novelty.
        """
        # 1. JEPA energy vs last latent state. THIS IS NOW A LIVE CALL.
        energy, _ = self.jepa.verify_state_transition(self.last_vec, new_vec)
        
        # 2. Cosine distance vs everything in "autonomous_research"
        neighbors = await self.store.query(
            vector=new_vec.tolist(),
            namespace="autonomous_research",
            top_k=5,
            include_values=True
        )
        
        max_sim = 0
        if neighbors and neighbors.get('matches'):
            # Pinecone returns similarity scores directly
            max_sim = max(match['score'] for match in neighbors['matches'])
            
        novelty = 1 - max_sim
        
        # Weighted surprise score
        surprise = (0.7 * energy) + (0.3 * novelty)
        
        # Convert numpy float to native Python float for logging
        surprise_float = float(surprise)
        logger.info(f"Surprise score: {surprise_float:.4f} (Energy: {float(energy):.4f}, Novelty: {novelty:.4f})")
        
        # Hash the vector to cache it. Use tobytes for a hashable representation.
        vec_hash = hash(new_vec.tobytes())
        
        if surprise < self.novelty_threshold or vec_hash in self.cache:
            self.last_vec = new_vec # Update state regardless
            return 0.0
            
        self.cache[vec_hash] = True
        self.last_vec = new_vec
        
        return surprise_float

