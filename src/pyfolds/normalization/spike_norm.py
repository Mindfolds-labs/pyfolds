from __future__ import annotations

import torch
from torch import nn


class SpikeLayerNorm(nn.Module):
    """Optional normalization layer for spike tensors."""

    def __init__(self, features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(features, eps=eps, elementwise_affine=affine)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        if spikes.ndim < 2:
            raise ValueError("spikes tensor must have at least 2 dimensions")
        return self.norm(spikes)
