from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
from torch import nn


@dataclass
class ModelWaveConfig:
    input_dim: int = 28 * 28
    hidden_dim: int = 128
    num_classes: int = 10
    timesteps: int = 4


class ModelWave(nn.Module):
    """Variante wave com integração temporal simplificada."""

    def __init__(self, config: ModelWaveConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelWaveConfig()
        self.encoder = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_classes)

    def forward(self, x: torch.Tensor, state: dict[str, Any] | None = None):
        batch = x.size(0)
        flat = x.view(batch, -1)
        h = torch.tanh(self.encoder(flat))

        mem = state.get("mem") if state else None
        if mem is None:
            mem = torch.zeros_like(h)

        spikes_total = torch.zeros_like(h)
        for _ in range(self.config.timesteps):
            mem = 0.9 * mem + h
            spikes = (mem > 0.5).float()
            mem = mem * (1.0 - spikes)
            spikes_total += spikes

        features = spikes_total / float(self.config.timesteps)
        logits = self.classifier(features)

        out = {
            "spikes": spikes_total,
            "spike_rate": float(features.mean().item()),
            "v_mean": float(mem.mean().item()),
            "state": {"mem": mem},
        }
        return logits, out

    def init_state(self, batch_size: int, device: str | torch.device):
        return {"mem": torch.zeros(batch_size, self.config.hidden_dim, device=device)}

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
        self.config = ModelWaveConfig(**cfg)
