"""Modelo principal FOLDSNet."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .connections import _create_sparse_connections
from .layers import _create_it, _create_lgn, _create_retina, _create_v1
from .serialization import load_payload, save_payload


class FOLDSNet(nn.Module):
    """Rede hierárquica Retina→LGN→V1→IT com neurônios MPJRD."""

    _VARIANTS = {
        "2L": {"retina": 8, "lgn": 8, "v1": 16, "it": 8},
        "4L": {"retina": 49, "lgn": 49, "v1": 98, "it": 49},
        "5L": {"retina": 64, "lgn": 64, "v1": 128, "it": 64},
        "6L": {"retina": 128, "lgn": 128, "v1": 256, "it": 128},
    }
    _RETINA_DENDRITES = 4
    _RETINA_SYNAPSES = 4
    _CORTICAL_DENDRITES = 4
    _CORTICAL_SYNAPSES = 8
    _AGGREGATION_TEMPERATURE = 0.8

    def __init__(self, input_shape: tuple[int, int, int], n_classes: int, variant: str = "4L"):
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError(f"Variante inválida '{variant}'. Use: {list(self._VARIANTS)}")

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

        self._validate_layer_profiles()

        lgn_to_v1, v1_to_it = _create_sparse_connections(self.n_lgn, self.n_v1, self.n_it)
        self.register_buffer("lgn_to_v1", lgn_to_v1)
        self.register_buffer("v1_to_it", v1_to_it)
        self.register_buffer("lgn_to_v1_exists", (lgn_to_v1.sum(dim=1, keepdim=True) > 0))
        self.register_buffer("v1_to_it_exists", (v1_to_it.sum(dim=1, keepdim=True) > 0))
        # register_buffer: migra automaticamente com .to(device)
        self.register_buffer("pixel_map", self._build_pixel_map())

        self.classifier = nn.Linear(self.n_it, n_classes)

    def _validate_layer_profiles(self) -> None:
        """Valida coerência entre perfis biológicos e entrada dendrítica esperada."""

        def _cfg_attr(layer: nn.ModuleList, attr: str, expected: int, stage: str) -> None:
            neuron = layer[0]
            cfg = getattr(neuron, "cfg", None)
            if cfg is None:
                return
            value = getattr(cfg, attr, None)
            if value != expected:
                raise ValueError(
                    f"Perfil inválido em '{stage}': {attr}={value}, esperado={expected}. "
                    "Verifique n_dendrites/n_synapses_per_dendrite da camada."
                )

        _cfg_attr(self.retina, "n_dendrites", self._RETINA_DENDRITES, "retina")
        _cfg_attr(self.retina, "n_synapses_per_dendrite", self._RETINA_SYNAPSES, "retina")
        _cfg_attr(self.v1, "n_dendrites", self._CORTICAL_DENDRITES, "v1")
        _cfg_attr(self.v1, "n_synapses_per_dendrite", self._CORTICAL_SYNAPSES, "v1")
        _cfg_attr(self.it, "n_dendrites", self._CORTICAL_DENDRITES, "it")
        _cfg_attr(self.it, "n_synapses_per_dendrite", self._CORTICAL_SYNAPSES, "it")

    def _build_pixel_map(self) -> torch.Tensor:
        """Cria mapa Retina preservando localidade espacial (patches 4×4 por neurônio)."""
        channels, height, width = self.input_shape
        if height < 4 or width < 4:
            raise ValueError(
                f"Input shape {self.input_shape} inválido: altura/largura mínimas são 4 para Retina 4×4."
            )

        side = max(1, int(self.n_retina**0.5))
        y_positions = torch.linspace(0, max(0, height - 1), steps=side).round().long()
        x_positions = torch.linspace(0, max(0, width - 1), steps=side).round().long()
        centers = torch.cartesian_prod(y_positions, x_positions)

        needed = self.n_retina
        if centers.shape[0] < needed:
            repeats = (needed + centers.shape[0] - 1) // centers.shape[0]
            centers = centers.repeat(repeats, 1)
        centers = centers[:needed]

        offsets = torch.tensor([-1, 0, 1, 2], dtype=torch.long)
        base_indices = []
        for idx, (cy, cx) in enumerate(centers):
            ys = (cy + offsets).clamp(0, height - 1)
            xs = (cx + offsets).clamp(0, width - 1)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            channel = idx % channels
            flat = channel * (height * width) + grid_y.reshape(-1) * width + grid_x.reshape(-1)
            base_indices.append(flat)
        return torch.stack(base_indices, dim=0)

    @staticmethod
    def _sparse_activity_pool(inputs: torch.Tensor, mask: torch.Tensor, has_edges: torch.Tensor, temperature: float) -> torch.Tensor:
        """Agrega conexões esparsas com softmax mascarado (mais expressivo que média simples)."""
        effective_mask = torch.where(has_edges, mask > 0, torch.ones_like(mask, dtype=torch.bool))
        logits = inputs.unsqueeze(1) / max(temperature, 1e-6)
        logits = logits.masked_fill(~effective_mask.unsqueeze(0), float("-inf"))
        weights = F.softmax(logits, dim=-1)
        return (weights * inputs.unsqueeze(1)).sum(dim=-1)

    def _validate_input_shape(self, x: torch.Tensor) -> None:
        expected = tuple(self.input_shape)
        got = tuple(x.shape[1:])
        if got != expected:
            raise ValueError(
                f"Shape de entrada incompatível para FOLDSNet {self.variant}: esperado={expected}, recebido={got}."
            )

    def forward(self, x: torch.Tensor, *, return_intermediates: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass hierárquico Retina → LGN → V1 → IT → classificador."""
        self._validate_input_shape(x)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        retina_pixels = x_flat[:, self.pixel_map]
        retina_dendrites = retina_pixels.view(batch_size, self.n_retina, self._RETINA_DENDRITES, self._RETINA_SYNAPSES)

        r1 = []
        for i, neuron in enumerate(self.retina):
            out = neuron(retina_dendrites[:, i])
            r1.append(out["spikes"])
        r1 = torch.stack(r1, dim=1)

        r2 = []
        for i, neuron in enumerate(self.lgn):
            lgn_input = r1[:, i].view(batch_size, 1, 1).repeat(1, 4, 4)
            out = neuron(lgn_input)
            r2.append(out["spikes"])
        r2 = torch.stack(r2, dim=1)

        r3 = []
        v1_agg = self._sparse_activity_pool(r2, self.lgn_to_v1, self.lgn_to_v1_exists, self._AGGREGATION_TEMPERATURE)
        for i, neuron in enumerate(self.v1):
            v1_input = v1_agg[:, i].view(batch_size, 1, 1).repeat(1, self._CORTICAL_DENDRITES, self._CORTICAL_SYNAPSES)
            out = neuron(v1_input)
            r3.append(out["spikes"])
        r3 = torch.stack(r3, dim=1)

        r4 = []
        it_agg = self._sparse_activity_pool(r3, self.v1_to_it, self.v1_to_it_exists, self._AGGREGATION_TEMPERATURE)
        for i, neuron in enumerate(self.it):
            it_input = it_agg[:, i].view(batch_size, 1, 1).repeat(1, self._CORTICAL_DENDRITES, self._CORTICAL_SYNAPSES)
            out = neuron(it_input)
            r4.append(out["spikes"])
        r4 = torch.stack(r4, dim=1)

        logits = self.classifier(r4)
        if return_intermediates:
            return {"retina": r1, "lgn": r2, "v1": r3, "it": r4, "logits": logits}
        return logits

    def save(self, path: str, fmt: str = "fold", include_metadata: bool = False, **kwargs) -> None:
        """Salva modelo em .fold ou .mind."""
        fmt = kwargs.pop("format", fmt)
        if kwargs:
            raise TypeError(f"Argumentos inesperados: {sorted(kwargs)}")
        payload: dict = {
            "state_dict": self.state_dict(),
            "input_shape": self.input_shape,
            "n_classes": self.n_classes,
            "variant": self.variant,
            "schema_version": 2,
        }
        if include_metadata:
            payload["metadata"] = {"format": fmt, "model": "FOLDSNet"}
        save_payload(path, fmt, payload)

    @classmethod
    def load(cls, path: str, fmt: str = "fold", device: str = "cpu", **kwargs) -> "FOLDSNet":
        """Carrega modelo salvo em .fold ou .mind."""
        fmt = kwargs.pop("format", fmt)
        if kwargs:
            raise TypeError(f"Argumentos inesperados: {sorted(kwargs)}")
        payload = load_payload(path, fmt, map_location=device)
        schema_version = int(payload.get("schema_version", 1))
        if schema_version not in {1, 2}:
            raise ValueError(f"Schema de serialização FOLDSNet não suportado: {schema_version}")
        model = cls(
            input_shape=tuple(payload["input_shape"]),
            n_classes=payload["n_classes"],
            variant=payload["variant"],
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return model
