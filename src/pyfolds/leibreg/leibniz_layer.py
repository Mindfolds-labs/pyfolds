"""Feature projection into LEIBREG conceptual coordinates."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeibnizLayer(nn.Module):
    """Project arbitrary features into the LEIBREG space."""

    def __init__(
        self,
        dim_input: int,
        dim_output: int = 4,
        use_bias: bool = True,
        normalize_output: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if dim_input <= 0 or dim_output <= 0:
            raise ValueError("dim_input and dim_output must be > 0")

        self.proj = nn.Linear(dim_input, dim_output, bias=use_bias)
        self.norm = nn.LayerNorm(dim_output)
        self.normalize_output = normalize_output
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features into conceptual coordinates.

        Args:
            x: Input tensor with shape ``[..., dim_input]``.

        Returns:
            Tensor with shape ``[..., dim_output]``.
        """
        y = self.norm(self.proj(x))
        if self.normalize_output:
            y = F.normalize(y, p=2, dim=-1, eps=self.eps)
        return y
