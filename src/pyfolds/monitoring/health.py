"""Health checks para neurônios PyFolds."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class NeuronHealthCheck:
    """Monitora métricas de saúde e gera alertas de operação."""

    def __init__(self, neuron, thresholds: Dict[str, float] | None = None):
        self.neuron = neuron
        self.thresholds = thresholds or {
            "dead_neuron_rate": 0.05,
            "saturation_ratio": 0.30,
            "min_spike_rate": 0.01,
        }

    def check(self) -> Tuple[HealthStatus, List[str]]:
        metrics = self.neuron.get_metrics()
        alerts: List[str] = []
        status = HealthStatus.HEALTHY

        dead_ratio = metrics.get("dead_neuron_ratio", 0.0)
        if dead_ratio > self.thresholds["dead_neuron_rate"]:
            status = HealthStatus.CRITICAL
            alerts.append(f"Alta taxa de neurônios mortos: {dead_ratio:.2%}")

        saturation = metrics.get("saturation_ratio", 0.0)
        if saturation > self.thresholds["saturation_ratio"] and status != HealthStatus.CRITICAL:
            status = HealthStatus.DEGRADED
            alerts.append(f"Saturação alta: {saturation:.2%}")

        spike_rate = metrics.get("spike_rate", 1.0)
        if spike_rate < self.thresholds["min_spike_rate"] and status != HealthStatus.CRITICAL:
            status = HealthStatus.DEGRADED
            alerts.append(f"Taxa de disparo muito baixa: {spike_rate:.3f}")

        return status, alerts
