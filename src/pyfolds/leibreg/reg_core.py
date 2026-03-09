"""Geometric resonance reasoning core for LEIBREG.

Pairwise distance attention has O(seq^2) complexity.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from pyfolds.telemetry.types import TelemetryEvent


class ResonanceAttention(nn.Module):
    """Distance-kernel self-attention on tensors of shape ``[batch, seq, dim]``."""

    def __init__(self, dim: int, init_temperature: float = 0.1, eps: float = 1e-8, telemetry_collector: Optional[Any] = None) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.value = nn.Linear(dim, dim)
        self.log_temp = nn.Parameter(torch.tensor(float(init_temperature)).log())
        self.eps = eps
        self.telemetry_collector = telemetry_collector

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp_min(1e-5)

    def _emit(self, payload: dict[str, float]) -> None:
        if self.telemetry_collector is None:
            return
        try:
            self.telemetry_collector.emit(TelemetryEvent(0.0, "leibreg_resonance", "leibreg.reg_core", 0, payload))
        except Exception:
            return

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must be [batch, seq, dim]")
        dist = torch.cdist(x, x, p=2)
        temp = self.temperature.to(dtype=x.dtype, device=x.device)
        kernel = 1.0 / (1.0 + (dist / temp) ** 2)
        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError("mask must be [batch, seq]")
            valid = (mask.unsqueeze(1) & mask.unsqueeze(2)).to(dtype=kernel.dtype)
            kernel = kernel * valid
        attn = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        out = torch.matmul(attn, self.value(x))

        entropy = -(attn * (attn.clamp_min(self.eps)).log()).sum(dim=-1).mean()
        self._emit(
            {
                "resonance_temperature": float(temp.mean().item()),
                "distance_mean": float(dist.mean().item()),
                "attention_entropy": float(entropy.item()),
            }
        )
        return out


class REGBlock(nn.Module):
    """Pre-norm residual resonance block."""

    def __init__(self, dim: int, ff_mult: int = 4, telemetry_collector: Optional[Any] = None) -> None:
        super().__init__()
        hidden = dim * ff_mult
        self.attn = ResonanceAttention(dim=dim, telemetry_collector=telemetry_collector)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, mask=mask))
        x = self.norm2(x + self.ff(x))
        return x


class REGCore(nn.Module):
    """Stack of :class:`REGBlock` layers."""

    def __init__(self, dim: int = 4, depth: int = 2, telemetry_collector: Optional[Any] = None) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be > 0")
        self.blocks = nn.ModuleList([REGBlock(dim=dim, telemetry_collector=telemetry_collector) for _ in range(depth)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


# Backward compatibility alias.
ProximityAttention = ResonanceAttention
