"""
Path: brain/imagination_engine.py
Fast latent roll-outs with JEPA.
"""
import numpy as np
from typing import List
from brain.jepa_aligner import JEPAAligner

class ImaginationEngine:
    def __init__(self, jepa: JEPAAligner, horizon: int = 3):
        self.jepa = jepa
        self.H = horizon

    def imagine(self, start_vec: np.ndarray, actions: List[str]) -> List[str]:
        """
        Returns list of *predicted* latent abstracts (human readable stub).
        """
        z = start_vec
        preds = []
        for a in actions:
            z = np.dot(self.jepa.predictor_weights, z)  # latent transition
            preds.append(f"imagined_{a}: {z[:10]}")     # stub english
        return preds
