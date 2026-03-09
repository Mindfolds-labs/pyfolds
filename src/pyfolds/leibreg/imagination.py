"""Imagination module with optional engram-memory retrieval."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from pyfolds.advanced.engram import EngramBank


class Imagination(nn.Module):
    """Either transform concept vectors or query an EngramBank."""

    def __init__(
        self,
        engram_bank: Optional[EngramBank] = None,
        associative_memory: Optional[Any] = None,
        hidden_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.engram_bank = engram_bank
        self.associative_memory = associative_memory
        self.proj = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 4))
        self.conf_head = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())

    def transform(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.proj(q)
        conf = self.conf_head(y)
        return y, conf

    def retrieve(self, q: torch.Tensor, k: int = 10, area: Optional[str] = None, query_phase: Optional[float] = None) -> Dict[str, Any]:
        if self.engram_bank is None:
            return {"concepts": [], "experiences": [], "scores": torch.empty(0), "backend": "missing"}
        query = q.detach().float().flatten().cpu()
        matches = self.engram_bank.search_by_resonance(query_pattern=query, query_phase=query_phase, area=area, top_k=k)
        concepts = [m.concept for m in matches]
        experiences = [m.to_dict() for m in matches]
        if matches:
            mem = torch.stack([m.wave_pattern.float() for m in matches], dim=0)
            qn = query / query.norm(p=2).clamp_min(1e-8)
            mn = mem / mem.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            scores = torch.matmul(mn, qn)
        else:
            scores = torch.empty(0)
        return {"concepts": concepts, "experiences": experiences, "scores": scores, "backend": "engram_bank"}

    def forward(self, q: torch.Tensor, k: int = 10, area: Optional[str] = None, query_phase: Optional[float] = None):
        if self.engram_bank is not None:
            return self.retrieve(q, k=k, area=area, query_phase=query_phase)
        return self.transform(q)
