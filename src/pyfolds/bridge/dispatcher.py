"""Bridge de exportação agnóstica do core PyFolds."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import torch


class MindDispatcher:
    """Interface agnóstica para exportação de dados do Core PyFolds."""

    @staticmethod
    def prepare_payload(
        layer_id: str,
        spikes: torch.Tensor | Any,
        weights: torch.Tensor | Any,
        health_score: float,
    ) -> dict[str, Any]:
        """Prepara pacote serializável para consumo externo."""
        spikes_value = (
            spikes.detach().cpu().tolist() if isinstance(spikes, torch.Tensor) else spikes
        )
        weights_mean = float(weights.mean()) if isinstance(weights, torch.Tensor) else 0.0
        return {
            "layer_id": layer_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {
                "spikes": spikes_value,
                "weights_mean": weights_mean,
                "health": health_score,
            },
        }

    @staticmethod
    def get_topology_map(network: Any) -> list[dict[str, Any]]:
        """Exporta estrutura em formato serializável sem dependências externas."""
        topology: list[dict[str, Any]] = []
        for i, layer in enumerate(getattr(network, "layers", [])):
            layer_cfg = getattr(layer, "config", getattr(layer, "cfg", None))
            if hasattr(layer_cfg, "to_dict"):
                cfg_payload = layer_cfg.to_dict()
            elif isinstance(layer_cfg, dict):
                cfg_payload = layer_cfg
            else:
                cfg_payload = {}
            topology.append({"layer_index": i, "config": cfg_payload})
        return topology
