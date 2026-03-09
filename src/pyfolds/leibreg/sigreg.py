"""Stability regularizers for conceptual embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Simple isotropy regularizer.

    Penalizes deviation between covariance and scaled identity.
    This is cheaper and more stable than Epps–Pulley for training loops.
    """

    def __init__(self, weight: float = 0.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = float(weight)
        self.eps = eps

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0.0:
            return q.new_zeros(())
        if q.numel() == 0:
            return q.new_zeros(())
        z = q.reshape(-1, q.shape[-1])
        if z.shape[0] < 2:
            return q.new_zeros(())
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / max(z.shape[0] - 1, 1)
        target = torch.eye(cov.shape[0], device=q.device, dtype=q.dtype) * (torch.trace(cov) / cov.shape[0]).clamp_min(self.eps)
        return self.weight * torch.mean((cov - target) ** 2)
