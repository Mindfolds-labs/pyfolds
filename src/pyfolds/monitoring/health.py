"""Health checks para neurônios PyFolds."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple

import torch


class HealthStatus(Enum):
    """Estados de saúde operacional calculados pelo monitoramento."""

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

        dead_ratio = float(metrics.get("dead_neuron_ratio", metrics.get("protection_ratio", 0.0)))
        if dead_ratio > self.thresholds["dead_neuron_rate"]:
            status = HealthStatus.CRITICAL
            alerts.append(f"Alta taxa de neurônios mortos: {dead_ratio:.2%}")

        saturation = metrics.get("saturation_ratio", 0.0)
        if saturation > self.thresholds["saturation_ratio"] and status != HealthStatus.CRITICAL:
            status = HealthStatus.DEGRADED
            alerts.append(f"Saturação alta: {saturation:.2%}")

        spike_rate = float(metrics.get("spike_rate", metrics.get("r_hat", 0.0)))
        if spike_rate < self.thresholds["min_spike_rate"] and status != HealthStatus.CRITICAL:
            status = HealthStatus.DEGRADED
            alerts.append(f"Taxa de disparo muito baixa: {spike_rate:.3f}")

        return status, alerts


class NeuronHealthMonitor:
    """Monitor de saúde contínuo para execução longa."""

    def __init__(self, neuron, check_every_n_steps: int = 100):
        self.neuron = neuron
        self.check_every = max(1, int(check_every_n_steps))
        self.step_count = 0
        self.alerts: List[str] = []

    def check_health(self) -> Dict[str, bool]:
        self.step_count += 1
        if self.step_count % self.check_every != 0:
            return {}

        issues: Dict[str, bool] = {}
        N = self.neuron.N
        I = self.neuron.I

        nan_in_n = bool(torch.isnan(N.float()).any().item())
        inf_in_n = bool(torch.isinf(N.float()).any().item())
        issues["nan_in_N"] = nan_in_n
        issues["inf_in_N"] = inf_in_n

        out_of_bounds = bool(((N < self.neuron.cfg.n_min) | (N > self.neuron.cfg.n_max)).any().item())
        issues["N_out_of_bounds"] = out_of_bounds

        theta_val = float(self.neuron.theta.item())
        theta_diverging = not (self.neuron.cfg.theta_min <= theta_val <= self.neuron.cfg.theta_max)
        issues["theta_diverging"] = theta_diverging

        i_unstable = abs(float(I.float().mean().item())) > 100.0
        issues["I_unstable"] = i_unstable

        collapsed = float(N.float().std().item()) < 0.1
        issues["collapsed_weights"] = collapsed

        for key, active in issues.items():
            if active:
                self.alerts.append(f"Step {self.step_count}: {key}")

        return {k: v for k, v in issues.items() if v}

    def get_health_score(self) -> float:
        if not self.alerts:
            return 100.0
        return max(0.0, 100.0 - len(self.alerts) * 10.0)
