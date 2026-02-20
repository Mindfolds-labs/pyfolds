"""Bridge de exportação agnóstica do core PyFolds."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import torch


class MindDispatcher:
    """Único ponto de saída de dados do PyFolds para o ecossistema Mind."""

    @staticmethod
    def capture_event(
        layer_id: str,
        spikes: torch.Tensor | Any,
        weights: torch.Tensor | Any,
        metrics: dict[str, Any] | Any,
    ) -> dict[str, Any]:
        """Prepara dados serializáveis para MindStream/MindAudis sem acoplamento."""
        event = {
            "origin": "pyfolds_v2.1.1",
            "layer": layer_id,
            "ts": datetime.now(UTC).isoformat(),
            "payload": {
                "activity": spikes.detach().cpu().tolist()
                if hasattr(spikes, "detach")
                else spikes,
                "weights_avg": float(weights.mean()) if hasattr(weights, "mean") else 0.0,
                "health": metrics,
            },
        }
        # Contrato legado (app atual/tests) mantido em paralelo ao payload canônico.
        event["layer_id"] = layer_id
        event["timestamp"] = event["ts"]
        return event

    @staticmethod
    def prepare_payload(
        layer_id: str,
        spikes: torch.Tensor | Any,
        weights: torch.Tensor | Any,
        health_score: float,
    ) -> dict[str, Any]:
        """Mantém contrato legado do dispatcher para consumidores atuais."""
        event = MindDispatcher.capture_event(
            layer_id=layer_id,
            spikes=spikes,
            weights=weights,
            metrics=health_score,
        )
        return {
            "layer_id": event["layer_id"],
            "timestamp": event["timestamp"],
            "data": {
                "spikes": event["payload"]["activity"],
                "weights_mean": event["payload"]["weights_avg"],
                "health": event["payload"]["health"],
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
