"""
Path: mind/differentiable_store.py
Soft top-k retrieval with Gumbel-Softmax so memory choice is learnable.
"""
import torch
import torch.nn as nn
from typing import Tuple
from pinecone import Pinecone

class DifferentiableStore(nn.Module):
    def __init__(self, index, namespace: str, top_k: int = 5, tau: float = 1.0):
        super().__init__()
        self.index = index
        self.ns    = namespace
        self.top_k = top_k
        self.tau   = tau  # Gumbel temperature

    def forward(self, query_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (soft_retrieved_vector, indices)  shape: [1024], [top_k]
        """
        res = self.index.query(namespace=self.ns, vector=query_vec.tolist(),
                               top_k=self.top_k, include_values=True)
        logits = torch.tensor([m['score'] for m in res['matches']])  # [top_k]
        vecs   = torch.tensor([m['values'] for m in res['matches']])  # [top_k, 1024]

        soft_mask = nn.functional.gumbel_softmax(logits, tau=self.tau, hard=False)
        soft_vec  = torch.matmul(soft_mask.unsqueeze(0), vecs).squeeze(0)  # [1024]
        return soft_vec, torch.arange(self.top_k)
