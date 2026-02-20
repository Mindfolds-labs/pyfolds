from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
from torch import nn


@dataclass
class ModelPyConfig:
    input_dim: int = 28 * 28
    hidden_dim: int = 128
    num_classes: int = 10


class ModelPy(nn.Module):
    """Modelo base simples para classificação MNIST."""

    def __init__(self, config: ModelPyConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelPyConfig()
        self.net = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.num_classes),
        )

    def forward(self, x: torch.Tensor, state: dict[str, Any] | None = None):
        x = x.view(x.size(0), -1)
        logits = self.net(x)
        out = {
            "spikes": torch.zeros(x.size(0), self.config.hidden_dim, device=x.device),
            "spike_rate": 0.0,
            "v_mean": 0.0,
            "state": state or {},
        }
        return logits, out

    def init_state(self, batch_size: int, device: str | torch.device):
        return {"dummy": torch.zeros(batch_size, 1, device=device)}

    def detach_state(self, state: dict[str, Any] | None):
        if not state:
            return {}
        detached: dict[str, Any] = {}
        for key, value in state.items():
            detached[key] = value.detach() if hasattr(value, "detach") else value
        return detached

    def get_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def load_config(self, cfg: dict[str, Any]) -> None:
        self.config = ModelPyConfig(**cfg)
