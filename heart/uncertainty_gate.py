"""
Path: heart/uncertainty_gate.py
Role: Gating mechanism based on uncertainty.
"""
from loguru import logger
import random

class UncertaintyGate:
    def __init__(self, reward_model):
        self.reward_model = reward_model

    def should_block(self, state_vec: list) -> tuple[bool, float]:
        """
        Determines if the content should be blocked based on uncertainty.
        Returns (should_block: bool, uncertainty_score: float)
        """
        # Placeholder logic:
        # In a real system, this would use the reward model or another model to estimate uncertainty.
        # For now, we return a low uncertainty to allow promotion if other checks pass,
        # or we could simulate it.
        # The prompt implies we check 'uncertainty > UNCERTAINTY_CUT' (0.35).

        # Let's calculate a dummy uncertainty based on the state vector to be deterministic but variable
        # For valid state_vec, uncertainty is low.
        if not state_vec or all(v == 0.0 for v in state_vec):
             return True, 1.0 # High uncertainty for empty/zero vector

        # We can use the reward model's prediction as a proxy for "certainty" in a way?
        # Or just return a safe value since this is a stub.
        uncertainty = 0.1
        return False, uncertainty
