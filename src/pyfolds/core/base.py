"""Interfaces base para componentes centrais do PyFolds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BasePlasticityRule(ABC):
    """Contrato mínimo para regras de plasticidade locais."""

    @abstractmethod
    def update(
        self,
        pre_rate: torch.Tensor,
        post_rate: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        """Retorna delta de pesos para o passo atual."""


class BaseNeuron(nn.Module, ABC):
    """Contrato comum para neurônios MPJRD e variantes."""

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Executa o passo forward do neurônio."""

    @abstractmethod
    def apply_plasticity(self, dt: float = 1.0, **kwargs: Any) -> None:
        """Aplica atualizações plásticas acumuladas."""

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas agregadas do estado interno."""
