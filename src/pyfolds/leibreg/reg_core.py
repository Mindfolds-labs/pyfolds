"""Proximity-based reasoning core for LEIBREG."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ProximityAttention(nn.Module):
    """Distance-kernel attention over ``[batch, seq, dim]`` tensors.

    Complexity is quadratic in sequence length due to pairwise distances.
    """

    def __init__(self, dim: int, kernel: str = "gaussian", temperature: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.value = nn.Linear(dim, dim)
        self.kernel = kernel
        self.temperature = max(float(temperature), 1e-4)
        self.eps = eps

    def _kernelize(self, dist: torch.Tensor) -> torch.Tensor:
        scaled = dist / self.temperature
        if self.kernel == "gaussian":
            weights = torch.exp(-(scaled**2))
        elif self.kernel == "inverse":
            weights = 1.0 / (1.0 + scaled)
        elif self.kernel == "cauchy":
            weights = 1.0 / (1.0 + scaled**2)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        return weights

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq, dim]")
        dist = torch.cdist(x, x, p=2)
        w = self._kernelize(dist)
        if mask is not None:
            # mask: [batch, seq], True=valid
            if mask.shape != x.shape[:2]:
                raise ValueError("mask must have shape [batch, seq]")
            pair_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).to(w.dtype)
            w = w * pair_mask
        denom = w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        attn = w / denom
        v = self.value(x)
        return torch.matmul(attn, v)


class REGBlock(nn.Module):
    """Residual proximity-attention block with feedforward network."""

    def __init__(self, dim: int, ff_mult: int = 4, kernel: str = "gaussian", temperature: float = 1.0) -> None:
        super().__init__()
        hidden = ff_mult * dim
        self.attn = ProximityAttention(dim=dim, kernel=kernel, temperature=temperature)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, mask=mask))
        x = self.norm2(x + self.ff(x))
        return x


class REGCore(nn.Module):
    """Stacked REG blocks."""

    def __init__(self, dim: int, depth: int = 2, kernel: str = "gaussian", temperature: float = 1.0) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be > 0")
        self.blocks = nn.ModuleList(
            [REGBlock(dim=dim, kernel=kernel, temperature=temperature) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask=mask)
        return x
