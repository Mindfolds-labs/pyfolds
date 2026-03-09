"""Stability regularizer for LEIBREG conceptual geometry."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F

from pyfolds.telemetry.types import TelemetryEvent


class SIGReg(nn.Module):
    """Projection-based isotropy/normality regularizer.

    Uses random unit projections and matches projected moments toward N(0,1).
    This approximates Epps-Pulley behavior at lower cost and improved stability.
    """

    def __init__(
        self,
        dim: int,
        num_projections: int = 32,
        weight: float = 1.0,
        enabled: bool = True,
        eps: float = 1e-8,
        telemetry_collector: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim
        self.weight = float(weight)
        self.enabled = enabled
        self.eps = eps
        self.telemetry_collector = telemetry_collector
        proj = torch.randn(num_projections, dim)
        proj = F.normalize(proj, p=2, dim=-1, eps=eps)
        self.register_buffer("projections", proj)

    def _emit(self, payload: dict[str, float]) -> None:
        if self.telemetry_collector is None:
            return
        try:
            self.telemetry_collector.emit(TelemetryEvent(0.0, "leibreg_sigreg", "leibreg.sigreg", 0, payload))
        except Exception:
            return

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if (not self.enabled) or self.weight <= 0.0 or q.numel() == 0:
            return q.new_zeros(())
        z = q.reshape(-1, q.shape[-1])
        if z.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {z.shape[-1]}")
        if z.shape[0] < 2:
            return q.new_zeros(())

        z = (z - z.mean(dim=0, keepdim=True)) / z.std(dim=0, keepdim=True).clamp_min(self.eps)
        p = torch.matmul(z, self.projections.T)
        mean_loss = (p.mean(dim=0) ** 2).mean()
        var = p.var(dim=0, unbiased=False)
        var_loss = ((var - 1.0) ** 2).mean()
        kurt = ((p**4).mean(dim=0) - 3.0) ** 2
        kurt_loss = kurt.mean()
        loss = self.weight * (mean_loss + var_loss + 0.1 * kurt_loss)

        self._emit({"sigreg_loss": float(loss.item()), "projection_variance_mean": float(var.mean().item())})
        return loss
