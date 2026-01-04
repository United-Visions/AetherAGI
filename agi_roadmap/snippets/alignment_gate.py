
snippets/alignment_gate.py


"""
Path: heart/uncertainty_gate.py
Rejects answers when epistemic uncertainty high.
"""
import torch
from heart.reward_model import RewardModel

class UncertaintyGate:
    def __init__(self, reward_model: RewardModel, threshold: float = 0.3):
        self.model = reward_model
        self.threshold = threshold

    def should_block(self, state_vec: list) -> tuple[bool, float]:
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state_vec)
            pred = self.model(x)            # single forward
            # simple uncertainty = |prediction|
            uncertainty = abs(pred.item())
            return uncertainty > self.threshold, uncertainty
snippets/self_mod.py (continued)


# same file as 06 – add alignment check before merge
align_ok, score = uncertainty_gate.should_block(state_vec)
if not align_ok:
    repo.git.checkout("main")
    repo.git.branch("-D", branch)
    return f"self-mod rejected – alignment uncertainty {score:.2f}"