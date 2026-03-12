"""Modelo principal FOLDSNet."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .connections import _create_sparse_connections
from .layers import _create_it, _create_lgn, _create_retina, _create_v1
from .serialization import load_foldsnet, save_foldsnet


class FOLDSNet(nn.Module):
    """Rede visual hierárquica biologicamente plausível baseada em MPJRD."""

    def __init__(self, input_shape: tuple[int, int, int], n_classes: int, variant: str = "4L"):
        super().__init__()
        if variant not in {"4L", "5L", "6L"}:
            raise ValueError("Variante inválida. Use '4L', '5L' ou '6L'.")

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.variant = variant

        channels, height, width = input_shape
        n_pixels = channels * height * width
        n_retina_base = max(1, n_pixels // 16)

        multiplier = {"4L": 1.0, "5L": 1.3, "6L": 1.7}[variant]
        n_retina = max(1, int(math.ceil(n_retina_base * multiplier)))
        n_lgn = n_retina
        n_v1 = n_retina * 2
        n_it = n_retina

        self.retina = _create_retina(n_retina)
        self.lgn = _create_lgn(n_lgn)
        self.v1 = _create_v1(n_v1)
        self.it = _create_it(n_it)

        lgn_to_v1, v1_to_it = _create_sparse_connections(n_lgn=n_lgn, n_v1=n_v1, n_it=n_it)
        self.register_buffer("lgn_to_v1", lgn_to_v1)
        self.register_buffer("v1_to_it", v1_to_it)

        pixel_map = torch.arange(n_pixels).repeat((n_retina * 16 // n_pixels) + 1)[: n_retina * 16]
        self.register_buffer("pixel_map", pixel_map.view(n_retina, 16))

        self.classifier = nn.Linear(n_it, n_classes)

    def get_init_kwargs(self) -> dict:
        """Retorna parâmetros necessários para reconstruir o modelo."""
        return {
            "input_shape": self.input_shape,
            "n_classes": self.n_classes,
            "variant": self.variant,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa o forward Retina → LGN → V1 → IT → classificador."""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        r1 = []
        for i, neuron in enumerate(self.retina):
            pixels = x_flat[:, self.pixel_map[i]]
            out = neuron(pixels.view(batch_size, 4, 4))
            r1.append(out["spikes"])
        r1 = torch.stack(r1, dim=1)

        r2 = []
        for i, neuron in enumerate(self.lgn):
            out = neuron(r1[:, i].view(batch_size, 1, 1).repeat(1, 4, 4))
            r2.append(out["spikes"])
        r2 = torch.stack(r2, dim=1)

        r3 = []
        for i, neuron in enumerate(self.v1):
            lgn_indices = torch.where(self.lgn_to_v1[i] > 0)[0]
            lgn_inputs = r2[:, lgn_indices].mean(dim=1)
            out = neuron(lgn_inputs.view(batch_size, 1, 1).repeat(1, 4, 8))
            r3.append(out["spikes"])
        r3 = torch.stack(r3, dim=1)

        r4 = []
        for i, neuron in enumerate(self.it):
            v1_indices = torch.where(self.v1_to_it[i] > 0)[0]
            v1_inputs = r3[:, v1_indices].mean(dim=1)
            out = neuron(v1_inputs.view(batch_size, 1, 1).repeat(1, 4, 8))
            r4.append(out["spikes"])
        r4 = torch.stack(r4, dim=1)

        return self.classifier(r4)

    def save(self, path: str, format: str = "fold", include_metadata: bool = False) -> None:
        """Salva modelo em .fold ou .mind."""
        metadata = self.get_init_kwargs() if include_metadata else {}
        save_foldsnet(self, path=path, fmt=format, metadata=metadata)

    @classmethod
    def load(cls, path: str, format: str = "fold", device: str = "cpu") -> "FOLDSNet":
        """Carrega modelo salvo em .fold ou .mind."""
        if not path.endswith(f".{format}"):
            raise ValueError("Extensão não corresponde ao formato solicitado.")
        model = load_foldsnet(path=path, model_cls=cls, device=device)
        model.eval()
        return model
