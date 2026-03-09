"""Imagination: expansão latente controlada no espaço conceitual."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class Imagination(nn.Module):
    """Gera hipótese latente refinada e score de confiança."""

    def __init__(self, *, hidden_dim: int = 16, dropout: float = 0.0, return_confidence: bool = True) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim deve ser > 0, recebido: {hidden_dim}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout deve estar em [0, 1), recebido: {dropout}")

        self.return_confidence = bool(return_confidence)
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.confidence_head = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())

    def _validate(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"Entrada deve ser torch.Tensor, recebido: {type(x)!r}")
        if x.shape[-1] != 4:
            raise ValueError(f"Última dimensão deve ser 4, recebido: {x.shape[-1]}")
        return x.float()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor | None]:
        """Retorna (hipótese_refinada, confiança_opcional)."""
        state = self._validate(x)
        delta = self.net(state)
        hypothesis = state + delta
        if not self.return_confidence:
            return hypothesis, None
        confidence = self.confidence_head(hypothesis)
        return hypothesis, confidence
