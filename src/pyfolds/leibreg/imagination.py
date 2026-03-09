"""Engram-backed memory recall adapter for LEIBREG."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from pyfolds.advanced.engram import EngramBank


class Imagination(nn.Module):
    """Recall memories from EngramBank using conceptual queries."""

    def __init__(self, engram_bank: Optional[EngramBank], associative_memory: Optional[Any] = None) -> None:
        super().__init__()
        self.engram_bank = engram_bank
        self.associative_memory = associative_memory

    def forward(self, q: torch.Tensor, k: int = 10, area: Optional[str] = None, query_phase: Optional[float] = None) -> Dict[str, Any]:
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
