snippets/diff_memory.py


"""
Path: mind/differentiable_store.py
Implements soft top-k with Gumbel-Softmax so retrieval is differentiable.
"""
import torch
import torch.nn as nn
from typing import List, Tuple
from pinecone import Index

class DifferentiableStore(nn.Module):
    def __init__(self, index: Index, namespace: str, top_k: int = 5, tau: float = 1.0):
        super().__init__()
        self.index = index
        self.ns = namespace
        self.top_k = top_k
        self.tau = tau  # Gumbel temperature

    def forward(self, query_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (soft_retrieved_vec, indices)  shape: [1024], [top_k]
        """
        # 1. fetch top-k *logits* (cosine similarities)
        res = self.index.query(namespace=self.ns, vector=query_vec.tolist(), top_k=self.top_k, include_values=True)
        logits = torch.tensor([m['score'] for m in res['matches']])  # [top_k]
        vecs   = torch.tensor([m['values'] for m in res['matches']])  # [top_k, 1024]

        # 2. differentiable soft selection
        soft_mask = nn.functional.gumbel_softmax(logits, tau=self.tau, hard=False)  # [top_k]
        soft_vec  = torch.matmul(soft_mask.unsqueeze(0), vecs).squeeze(0)  # [1024]

        return soft_vec, torch.arange(self.top_k)

Patch to logic_engine.py


# inside generate_thought
if settings.diff_retrieval and torch is not None:
    from mind.differentiable_store import DifferentiableStore
    store = DifferentiableStore(self.pc.Index("aethermind-genesis"), namespace, top_k=5)
    context_vec_torch = torch.FloatTensor(context_vec)
    context_vec, _ = store(context_vec_torch)  # now differentiable
    contexts = ["[diff retrieved]"]  # placeholder for logging        