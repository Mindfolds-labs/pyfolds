"""Geometric conceptual space for LEIBREG."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import WordSpaceOutput


class WordSpace(nn.Module):
    """Learnable concept embeddings with optional wave-aware modulation."""

    def __init__(
        self,
        concept_count: int,
        dim_base: int = 4,
        dim_context: int = 0,
        normalize_output: bool = True,
        wave_enabled: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if concept_count <= 0:
            raise ValueError("concept_count must be > 0")
        if dim_base <= 0:
            raise ValueError("dim_base must be > 0")
        if dim_context < 0:
            raise ValueError("dim_context must be >= 0")

        self.concept_count = concept_count
        self.dim_base = dim_base
        self.dim_context = dim_context
        self.normalize_output = normalize_output
        self.wave_enabled = wave_enabled
        self.eps = eps

        self.base_embedding = nn.Embedding(concept_count, dim_base)
        self.context_embedding = nn.Embedding(concept_count, dim_context) if dim_context > 0 else None
        nn.init.normal_(self.base_embedding.weight, mean=0.0, std=0.02)
        if self.context_embedding is not None:
            nn.init.normal_(self.context_embedding.weight, mean=0.0, std=0.02)

    def forward(self, concept_ids: torch.Tensor, wave_phase: Optional[torch.Tensor] = None) -> WordSpaceOutput:
        if concept_ids.numel() == 0:
            raise ValueError("concept_ids cannot be empty")
        if concept_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            raise TypeError("concept_ids must be an integer tensor")
        if torch.any(concept_ids < 0) or torch.any(concept_ids >= self.concept_count):
            raise ValueError("concept_ids contains out-of-range values")

        q_base = self.base_embedding(concept_ids.long())
        if self.wave_enabled and wave_phase is not None:
            q_base = q_base * (1.0 + 0.1 * torch.sin(wave_phase).to(q_base.device))

        q_context = self.context_embedding(concept_ids.long()) if self.context_embedding is not None else None
        q_total = torch.cat([q_base, q_context], dim=-1) if q_context is not None else q_base

        if self.normalize_output:
            q_total = F.normalize(q_total, p=2, dim=-1, eps=self.eps)

        return {
            "q_base": q_base,
            "q_context": q_context,
            "q_total": q_total,
            "dim_total": int(q_total.shape[-1]),
        }

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute L2 distance in conceptual space."""
        return torch.linalg.norm(a - b, dim=-1)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute stable cosine similarity in conceptual space."""
        return F.cosine_similarity(a, b, dim=-1, eps=self.eps)
