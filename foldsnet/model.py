"""Modelo principal FOLDSNet."""

from __future__ import annotations

import torch
from torch import nn

from .connections import _create_sparse_connections
from .layers import _create_it, _create_lgn, _create_retina, _create_v1
from .serialization import load_payload, save_payload


class FOLDSNet(nn.Module):
    """Rede hierárquica Retina→LGN→V1→IT com neurônios MPJRD."""

    _VARIANTS = {
        "4L": {"retina": 49, "lgn": 49, "v1": 98, "it": 49},
        "5L": {"retina": 64, "lgn": 64, "v1": 128, "it": 64},
        "6L": {"retina": 128, "lgn": 128, "v1": 256, "it": 128},
    }

    def __init__(self, input_shape: tuple[int, int, int], n_classes: int, variant: str = "4L"):
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError("Variante inválida. Use '4L', '5L' ou '6L'.")

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.variant = variant

        sizes = self._VARIANTS[variant]
        self.n_retina = sizes["retina"]
        self.n_lgn = sizes["lgn"]
        self.n_v1 = sizes["v1"]
        self.n_it = sizes["it"]

        self.retina = _create_retina(self.n_retina)
        self.lgn = _create_lgn(self.n_lgn)
        self.v1 = _create_v1(self.n_v1)
        self.it = _create_it(self.n_it)

        lgn_to_v1, v1_to_it = _create_sparse_connections(self.n_lgn, self.n_v1, self.n_it)
        self.register_buffer("lgn_to_v1", lgn_to_v1)
        self.register_buffer("v1_to_it", v1_to_it)

        self.pixel_map = self._build_pixel_map()
        self.classifier = nn.Linear(self.n_it, n_classes)

    def _build_pixel_map(self) -> torch.Tensor:
        """Cria mapa de pixels 4x4 para cada neurônio da retina."""
        n_pixels = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        if n_pixels < 16:
            raise ValueError("A entrada precisa ter ao menos 16 pixels para a Retina.")

        indices = []
        stride = max(1, (n_pixels - 16) // max(1, self.n_retina - 1))
        for i in range(self.n_retina):
            start = min(i * stride, n_pixels - 16)
            indices.append(torch.arange(start, start + 16, dtype=torch.long))
        return torch.stack(indices, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa forward pass hierárquico da FOLDSNet."""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        r1 = []
        for i, neuron in enumerate(self.retina):
            pixels = x_flat[:, self.pixel_map[i]]
            dendrites = pixels.view(batch_size, 4, 4)
            out = neuron(dendrites)
            r1.append(out["spikes"])
        r1 = torch.stack(r1, dim=1)

        r2 = []
        for i, neuron in enumerate(self.lgn):
            lgn_input = r1[:, i].view(batch_size, 1, 1).repeat(1, 4, 4)
            out = neuron(lgn_input)
            r2.append(out["spikes"])
        r2 = torch.stack(r2, dim=1)

        r3 = []
        for i, neuron in enumerate(self.v1):
            lgn_indices = torch.where(self.lgn_to_v1[i] > 0)[0]
            lgn_inputs = r2[:, lgn_indices].mean(dim=1).view(batch_size, 1, 1).repeat(1, 4, 8)
            out = neuron(lgn_inputs)
            r3.append(out["spikes"])
        r3 = torch.stack(r3, dim=1)

        r4 = []
        for i, neuron in enumerate(self.it):
            v1_indices = torch.where(self.v1_to_it[i] > 0)[0]
            v1_inputs = r3[:, v1_indices].mean(dim=1).view(batch_size, 1, 1).repeat(1, 4, 8)
            out = neuron(v1_inputs)
            r4.append(out["spikes"])
        r4 = torch.stack(r4, dim=1)

        logits = self.classifier(r4)
        return logits

    def save(self, path: str, format: str = "fold", include_metadata: bool = False) -> None:
        """Salva modelo em .fold ou .mind."""
        payload = {
            "state_dict": self.state_dict(),
            "input_shape": self.input_shape,
            "n_classes": self.n_classes,
            "variant": self.variant,
        }
        if include_metadata:
            payload["metadata"] = {"format": format, "model": "FOLDSNet"}
        save_payload(path, format, payload)

    @classmethod
    def load(cls, path: str, format: str = "fold", device: str = "cpu") -> "FOLDSNet":
        """Carrega modelo salvo em .fold ou .mind."""
        payload = load_payload(path, format, map_location=device)
        model = cls(
            input_shape=tuple(payload["input_shape"]),
            n_classes=payload["n_classes"],
            variant=payload["variant"],
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return model
