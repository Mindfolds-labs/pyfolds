"""REGCore: refinamento geométrico por proximidade no WordSpace."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

MetricName = Literal["euclidean", "cosine"]


class REGCore(nn.Module):
    """Refina conceitos no espaço 4D usando vizinhança geométrica.

    Hipótese: o raciocínio relacional pode ser aproximado por difusão local
    ponderada por proximidade, evitando mecanismo de atenção completo.
    """

    def __init__(
        self,
        *,
        metric: MetricName = "euclidean",
        temperature: float = 1.0,
        residual: float = 0.5,
        num_steps: int = 1,
    ) -> None:
        super().__init__()
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("metric deve ser 'euclidean' ou 'cosine'.")
        if temperature <= 0:
            raise ValueError(f"temperature deve ser > 0, recebido: {temperature}")
        if not (0.0 <= residual <= 1.0):
            raise ValueError(f"residual deve estar em [0, 1], recebido: {residual}")
        if num_steps <= 0:
            raise ValueError(f"num_steps deve ser > 0, recebido: {num_steps}")

        self.metric = metric
        self.temperature = float(temperature)
        self.residual = float(residual)
        self.num_steps = int(num_steps)

    def _pairwise_scores(self, x: Tensor) -> Tensor:
        if self.metric == "euclidean":
            distances = torch.cdist(x, x, p=2)
            return -distances / self.temperature
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-8)
        similarities = torch.matmul(x_norm, x_norm.transpose(-1, -2))
        return similarities / self.temperature

    def _validate(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError(f"Entrada deve ser torch.Tensor, recebido: {type(x)!r}")
        if x.ndim != 3:
            raise ValueError(f"Entrada deve ter shape [batch, n_tokens, 4], recebido: {tuple(x.shape)}")
        if x.shape[-1] != 4:
            raise ValueError(f"Última dimensão deve ser 4 (WordSpace), recebido: {x.shape[-1]}")

    def forward(self, x: Tensor) -> Tensor:
        """Aplica refinamento iterativo e mistura residual."""
        self._validate(x)
        state = x.float()
        for _ in range(self.num_steps):
            scores = self._pairwise_scores(state)
            weights = torch.softmax(scores, dim=-1)
            neighborhood = torch.matmul(weights, state)
            state = (self.residual * state) + ((1.0 - self.residual) * neighborhood)
        return state
