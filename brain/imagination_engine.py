"""
Path: brain/imagination_engine.py
Roll out H-step action sequences in latent space for lookahead planning.
"""
import numpy as np
from typing import List
from brain.jepa_aligner import JEPAAligner

class ImaginationEngine:
    def __init__(self, jepa: JEPAAligner, horizon: int = 5):
        self.jepa = jepa
        self.H  = horizon

    def imagine(self, start_vec: np.ndarray, plan: List[str]) -> List[str]:
        """Return list of human-readable stubs for each imagined step."""
        z = start_vec
        preds = []
        for t, action in enumerate(plan):
            z = np.dot(self.jepa.predictor_weights, z)   # latent transition
            energy = float(self.jepa.compute_energy(z, z))  # self-similarity as confidence
            preds.append(f"Step{t+1} {action} (confidence={1-energy:.2f})")
        return preds

    def pick_best_plan(self, start_vec: np.ndarray, candidates: List[List[str]]) -> List[str]:
        """Return the plan with lowest total imagined energy (highest expected alignment)."""
        best_energy, best_plan = float('inf'), []
        for plan in candidates:
            z = start_vec
            total_e = 0.0
            for a in plan:
                z_pred  = np.dot(self.jepa.predictor_weights, z)
                total_e += self.jepa.compute_energy(z, z_pred)
                z = z_pred
            if total_e < best_energy:
                best_energy, best_plan = total_e, plan
        return best_plan
