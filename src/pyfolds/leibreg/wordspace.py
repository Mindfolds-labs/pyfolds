"""Geometric conceptual space for LEIBREG."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


class WordSpace(nn.Module):
    """Learnable concept points on a conceptual hypersphere.

    Args:
        concept_count: Number of discrete concept IDs.
        dim_base: Base conceptual dimension (default ``4``).
        dim_context: Optional learned context dimensions concatenated to base.
        text_dim: Optional raw text feature dimension for projection helpers.
        image_dim: Optional raw image feature dimension for projection helpers.
        memory_dim: Optional raw memory feature dimension for projection helpers.
        normalize_output: If ``True``, forward output is L2-normalized.
        wave_enabled: Enables phase rotation pathway.
        telemetry_collector: Optional :class:`TelemetryCollector` compatible instance.
        eps: Numerical epsilon for normalization/division safety.
    """

    def __init__(
        self,
        concept_count: int = 1024,
        dim_base: int = 4,
        dim_context: int = 0,
        text_dim: Optional[int] = None,
        image_dim: Optional[int] = None,
        memory_dim: Optional[int] = None,
        normalize_output: bool = True,
        wave_enabled: bool = True,
        telemetry_collector: Optional[Any] = None,
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
        self.dim_total = dim_base + dim_context
        self.normalize_output = normalize_output
        self.wave_enabled = wave_enabled
        self.telemetry_collector = telemetry_collector
        self.eps = eps

        self.base_embedding = nn.Embedding(concept_count, dim_base)
        self.context_embedding = nn.Embedding(concept_count, dim_context) if dim_context > 0 else None

        # Orthogonal init is valid for 2D tables; rows are orthonormal when rows <= cols,
        # otherwise columns are orthonormal. This is still a stable geometric start.
        nn.init.orthogonal_(self.base_embedding.weight)
        if self.context_embedding is not None:
            nn.init.orthogonal_(self.context_embedding.weight)

        self.text_projector = nn.Linear(text_dim, dim_base) if text_dim is not None else None
        self.image_projector = nn.Linear(image_dim, dim_base) if image_dim is not None else None
        self.memory_projector = nn.Linear(memory_dim, dim_base) if memory_dim is not None else None

    def _emit(self, event_type: str, payload: Dict[str, float]) -> None:
        collector = self.telemetry_collector
        if collector is None:
            return
        event_cls = getattr(__import__("pyfolds.telemetry.types", fromlist=["TelemetryEvent"]), "TelemetryEvent", None)
        if event_cls is None:
            return
        try:
            collector.emit(event_cls(timestamp=0.0, event_type=event_type, source="leibreg.wordspace", step=0, payload=payload))
        except Exception:
            return

    def _apply_wave_rotation(self, q: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Rotate 4D vectors using paired Givens rotations.

        Rotates in planes ``(0,1)`` and ``(2,3)``. ``phase`` can be scalar,
        ``[..., 1]`` or broadcastable to ``q[..., 0]``.
        """
        if q.shape[-1] != 4:
            raise ValueError("Wave rotation requires dim_base == 4")
        if phase.ndim == 0:
            phase = phase.view(1)
        phase = phase.to(dtype=q.dtype, device=q.device)
        phase = torch.broadcast_to(phase, q.shape[:-1])
        c = torch.cos(phase)
        s = torch.sin(phase)

        q0, q1, q2, q3 = q.unbind(dim=-1)
        r0 = c * q0 - s * q1
        r1 = s * q0 + c * q1
        r2 = c * q2 - s * q3
        r3 = s * q2 + c * q3
        return torch.stack((r0, r1, r2, r3), dim=-1)

    def forward(
        self,
        concept_ids: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        wave_phase: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Lookup concept embeddings.

        Args:
            concept_ids: Integer tensor ``[...,]`` with values in
                ``[0, concept_count)``.
            phase: Optional phase tensor broadcastable to ``concept_ids``.
            context: Optional extra context features broadcastable to
                ``[..., dim_context]``.
            wave_phase: Backward-compatible alias for ``phase``.
        """
        if concept_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("concept_ids must be integer dtype")
        if concept_ids.numel() == 0:
            raise ValueError("concept_ids cannot be empty")
        if torch.any(concept_ids < 0) or torch.any(concept_ids >= self.concept_count):
            raise ValueError("concept_ids contains out-of-range values")

        phase_in = phase if phase is not None else wave_phase
        q_base = self.base_embedding(concept_ids.long())
        q_base = F.normalize(q_base, p=2, dim=-1, eps=self.eps)
        phase_applied = torch.zeros((), dtype=torch.bool, device=q_base.device)
        if self.wave_enabled and phase_in is not None and self.dim_base == 4:
            q_base = self._apply_wave_rotation(q_base, phase_in)
            phase_applied = torch.ones((), dtype=torch.bool, device=q_base.device)

        q_context = self.context_embedding(concept_ids.long()) if self.context_embedding is not None else None
        if q_context is not None and context is not None:
            q_context = q_context + context.to(dtype=q_context.dtype, device=q_context.device)
        q_total = torch.cat([q_base, q_context], dim=-1) if q_context is not None else q_base
        if self.normalize_output:
            q_total = F.normalize(q_total, p=2, dim=-1, eps=self.eps)

        norm = torch.linalg.vector_norm(q_total, dim=-1)
        self._emit("leibreg_wordspace", {"concept_norm_mean": float(norm.mean().item())})
        return {"q_base": q_base, "q_total": q_total, "phase_applied": phase_applied, "norm": norm}

    def project(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        projector = {
            "text": self.text_projector,
            "image": self.image_projector,
            "memory": self.memory_projector,
        }.get(modality)
        if projector is None:
            raise ValueError(f"Invalid modality: {modality}")
        out = projector(x)
        return F.normalize(out, p=2, dim=-1, eps=self.eps)

    def project_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x, "text")

    def project_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x, "image")

    def project_memory(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x, "memory")

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(a - b, dim=-1)

    def similarity(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        dist = self.distance(a, b)
        return 1.0 / (1.0 + dist.clamp_min(eps))
