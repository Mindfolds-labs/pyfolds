from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from pyfolds.utils.types import LearningMode


class TrainableModel(Protocol):
    def forward(self, x: torch.Tensor, **kwargs: Any) -> Any: ...

    def get_config(self) -> dict[str, Any]: ...

    def set_mode(self, mode: LearningMode) -> None: ...

    def sleep(self, duration: float) -> None: ...


@dataclass
class ModelMetadata:
    family: str
    config: dict[str, Any]
    supports_learning_mode: bool
    supports_sleep: bool


class FOLDSNetAdapter(nn.Module):
    """Adapter para alinhar FOLDSNet ao contrato padrão de treino."""

    def __init__(self, model: nn.Module, variant: str, dataset: str):
        super().__init__()
        self.model = model
        self._variant = variant
        self._dataset = dataset

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Any:
        return self.model(x)

    def get_config(self) -> dict[str, Any]:
        return {"variant": self._variant, "dataset": self._dataset}

    def set_mode(self, mode: LearningMode) -> None:
        if mode == LearningMode.INFERENCE:
            self.eval()
        else:
            self.train()


class MPJRDWrapper(nn.Module):
    """Wrapper para adaptar MNIST [B,1,28,28] para entrada MPJRD [B,D,S]."""

    def __init__(self, neuron: nn.Module, n_dendrites: int, n_synapses_per_dendrite: int):
        super().__init__()
        self.neuron = neuron
        self.n_dendrites = n_dendrites
        self.n_synapses_per_dendrite = n_synapses_per_dendrite

        hidden_dim = n_dendrites * n_synapses_per_dendrite
        self.proj = nn.Linear(784, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 10)
        nn.init.xavier_uniform_(self.proj.weight, gain=2.0)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        bsz = x.shape[0]
        x_flat = x.view(bsz, -1)
        x_proj = self.proj(x_flat)
        x_reshaped = x_proj.view(bsz, self.n_dendrites, self.n_synapses_per_dendrite)
        neuron_out = self.neuron(x_reshaped, **kwargs)

        if isinstance(neuron_out, dict):
            features = neuron_out.get("spikes", neuron_out.get("u", x_reshaped.mean(dim=(1, 2))))
        else:
            features = x_reshaped.mean(dim=(1, 2))

        if features.dim() > 2:
            features = features.view(bsz, -1)
        elif features.dim() == 1:
            features = features.unsqueeze(1)

        expected_dim = self.n_dendrites * self.n_synapses_per_dendrite
        if features.shape[1] != expected_dim:
            features = x_reshaped.view(bsz, -1)

        logits = self.classifier(features)
        return logits, neuron_out

    def get_config(self) -> dict[str, Any]:
        return self.neuron.get_config() if hasattr(self.neuron, "get_config") else {}

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.neuron, name)
