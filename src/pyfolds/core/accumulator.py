"""Acumulador de estatísticas para batch learning com modo denso/sparse-masked."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class AccumulatedStats:
    """Container para estatísticas acumuladas."""

    x_mean: Optional[torch.Tensor] = None
    gated_mean: Optional[torch.Tensor] = None
    post_rate: Optional[float] = None
    v_dend_mean: Optional[torch.Tensor] = None
    u_mean: Optional[float] = None
    theta_mean: Optional[float] = None
    r_hat_mean: Optional[float] = None
    adaptation_mean: Optional[float] = None
    spike_count: int = 0
    total_samples: int = 0
    sparsity: Optional[float] = None

    def is_valid(self) -> bool:
        return self.total_samples > 0

    def __repr__(self) -> str:
        if not self.is_valid():
            return "AccumulatedStats(empty)"
        return (
            f"AccumulatedStats(samples={self.total_samples}, "
            f"spikes={self.spike_count}, rate={self.post_rate:.3f})"
        )


class StatisticsAccumulator(nn.Module):
    """Acumulador de estatísticas com baseline denso e caminho sparse mascarado."""

    def __init__(
        self,
        n_dendrites: int,
        n_synapses: int,
        eps: float = 1e-8,
        track_extra: bool = False,
        max_history_len: int = 10000,
        *,
        mode: str = "dense",
        activity_threshold: float = 0.01,
        sparse_min_activity_ratio: float = 0.15,
        scientific_debug_stats: bool = False,
        enable_profiling: bool = False,
    ):
        super().__init__()

        if mode not in {"dense", "sparse_masked"}:
            raise ValueError("mode deve ser 'dense' ou 'sparse_masked'")
        if activity_threshold < 0:
            raise ValueError("activity_threshold deve ser >= 0")
        if not 0.0 <= sparse_min_activity_ratio <= 1.0:
            raise ValueError("sparse_min_activity_ratio deve estar em [0, 1]")

        self.n_dendrites = n_dendrites
        self.n_synapses = n_synapses
        self.eps = eps
        self.track_extra = track_extra
        self.max_history_len = max_history_len
        self.mode = mode
        self.activity_threshold = float(activity_threshold)
        self.sparse_min_activity_ratio = float(sparse_min_activity_ratio)
        self.scientific_debug_stats = scientific_debug_stats
        self.enable_profiling = enable_profiling

        self.register_buffer("acc_x", torch.zeros(n_dendrites, n_synapses))
        self.register_buffer("acc_gated", torch.zeros(n_dendrites))
        self.register_buffer("synapse_sample_count", torch.zeros(n_dendrites, n_synapses))
        self.register_buffer("acc_spikes", torch.zeros(1, dtype=torch.long))
        self.register_buffer("acc_count", torch.zeros(1, dtype=torch.long))

        if track_extra:
            self.register_buffer("acc_v_dend", torch.zeros(n_dendrites))
            self.register_buffer("acc_u", torch.zeros(1))
            self.register_buffer("acc_theta", torch.zeros(1))
            self.register_buffer("acc_r_hat", torch.zeros(1))
            self.register_buffer("acc_adaptation", torch.zeros(1))

        self.register_buffer("initialized", torch.tensor(False))

        self.last_activity_ratio: float = 0.0
        self.sparse_path_used: bool = False
        self.dense_fallback_used: bool = False
        self.last_accumulator_time_ms: float = 0.0
        self.nonzero_sample_ratio: float = 0.0

        self._history: Dict[str, Deque[float]] = {}
        self._history_enabled = False
        self._lock = Lock()

    @property
    def history(self) -> Dict[str, Deque[float]]:
        if not self._history_enabled:
            return {}
        return self._history

    @property
    def telemetry_snapshot(self) -> Dict[str, float | bool]:
        return {
            "accumulator_time_ms": float(self.last_accumulator_time_ms),
            "activity_ratio": float(self.last_activity_ratio),
            "sparse_path_used": bool(self.sparse_path_used),
            "dense_fallback_used": bool(self.dense_fallback_used),
            "nonzero_sample_ratio": float(self.nonzero_sample_ratio),
        }

    def enable_history(self, enable: bool = True) -> None:
        self._history_enabled = enable
        if enable and not self._history:
            self._history = {
                "spike_rate": deque(maxlen=self.max_history_len),
                "sparsity": deque(maxlen=self.max_history_len),
                "theta": deque(maxlen=self.max_history_len),
                "r_hat": deque(maxlen=self.max_history_len),
            }

    def reset(self) -> None:
        with self._lock:
            self.acc_x.zero_()
            self.acc_gated.zero_()
            self.synapse_sample_count.zero_()
            self.acc_spikes.zero_()
            self.acc_count.zero_()

            if self.track_extra:
                self.acc_v_dend.zero_()
                self.acc_u.zero_()
                self.acc_theta.zero_()
                self.acc_r_hat.zero_()
                self.acc_adaptation.zero_()

            self.initialized.fill_(False)
            self.last_activity_ratio = 0.0
            self.sparse_path_used = False
            self.dense_fallback_used = False
            self.last_accumulator_time_ms = 0.0
            self.nonzero_sample_ratio = 0.0

            if self._history_enabled:
                for key in self._history:
                    self._history[key].clear()

    def _accumulate_dense(self, x: torch.Tensor, gated: torch.Tensor) -> None:
        self.acc_x += x.sum(dim=0)
        self.acc_gated += gated.sum(dim=0)
        self.synapse_sample_count += torch.ones_like(x).sum(dim=0)

    def _accumulate_sparse_masked(self, x: torch.Tensor, gated: torch.Tensor) -> None:
        active_mask = x.abs() > self.activity_threshold
        self.last_activity_ratio = float(active_mask.float().mean().item())

        if self.last_activity_ratio > self.sparse_min_activity_ratio:
            self.dense_fallback_used = True
            self.sparse_path_used = False
            self._accumulate_dense(x, gated)
            return

        self.dense_fallback_used = False
        self.sparse_path_used = True

        active_mask_f = active_mask.to(dtype=x.dtype)
        self.acc_x += (x * active_mask_f).sum(dim=0)
        self.synapse_sample_count += active_mask_f.sum(dim=0)

        active_gated = active_mask.any(dim=2).to(dtype=gated.dtype)
        self.acc_gated += (gated * active_gated).sum(dim=0)

    def accumulate(
        self,
        x: torch.Tensor,
        gated: torch.Tensor,
        spikes: torch.Tensor,
        v_dend: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        r_hat: Optional[torch.Tensor] = None,
        adaptation: Optional[torch.Tensor] = None,
    ) -> None:
        batch_size = x.shape[0]

        if x.shape[1:] != (self.n_dendrites, self.n_synapses):
            raise ValueError(
                f"Esperado [B, {self.n_dendrites}, {self.n_synapses}], recebido {x.shape}"
            )
        if gated.dim() != 2 or gated.shape != (batch_size, self.n_dendrites):
            raise ValueError(f"gated deve ser [B, {self.n_dendrites}], recebido {gated.shape}")
        if spikes.dim() != 1 or spikes.shape[0] != batch_size:
            raise ValueError(f"spikes deve ser [B], recebido {spikes.shape}")

        x = x.detach()
        gated = gated.detach()
        spikes = spikes.detach()

        if self.acc_x.device != x.device:
            self.to(x.device)

        t0 = time.perf_counter() if self.enable_profiling else 0.0
        with self._lock:
            self.last_activity_ratio = 1.0
            self.sparse_path_used = False
            self.dense_fallback_used = False

            if self.mode == "dense":
                self._accumulate_dense(x, gated)
            else:
                self._accumulate_sparse_masked(x, gated)

            self.acc_spikes += spikes.sum().long()
            self.acc_count += batch_size
            self.synapse_sample_count.clamp_(min=0.0)

            if self.track_extra:
                if v_dend is not None:
                    self.acc_v_dend += v_dend.detach().sum(dim=0)
                if u is not None:
                    self.acc_u += u.detach().sum()
                if theta is not None:
                    theta_d = theta.detach()
                    self.acc_theta += theta_d.sum() if theta_d.dim() > 0 else theta_d * batch_size
                if r_hat is not None:
                    r_hat_d = r_hat.detach()
                    self.acc_r_hat += r_hat_d.sum() if r_hat_d.dim() > 0 else r_hat_d * batch_size
                if adaptation is not None:
                    adaptation_d = adaptation.detach()
                    self.acc_adaptation += (
                        adaptation_d.sum() if adaptation_d.dim() > 0 else adaptation_d * batch_size
                    )

            self.initialized.fill_(True)
            self.nonzero_sample_ratio = float((self.synapse_sample_count > 0).float().mean().item())

            if self._history_enabled:
                self._update_history(spikes, gated, theta, r_hat)

        if self.enable_profiling:
            self.last_accumulator_time_ms = (time.perf_counter() - t0) * 1000.0

    def _update_history(
        self,
        spikes: torch.Tensor,
        gated: torch.Tensor,
        theta: Optional[torch.Tensor],
        r_hat: Optional[torch.Tensor],
    ) -> None:
        if not self._history_enabled:
            return

        self._history["spike_rate"].append(spikes.float().mean().item())
        self._history["sparsity"].append((gated > 0).float().mean().item())

        if theta is not None:
            theta_mean = theta.mean().item() if theta.dim() > 0 else float(theta.item())
            self._history["theta"].append(theta_mean)

        if r_hat is not None:
            r_hat_mean = r_hat.mean().item() if r_hat.dim() > 0 else float(r_hat.item())
            self._history["r_hat"].append(r_hat_mean)

    def get_averages(self) -> AccumulatedStats:
        with self._lock:
            if self.acc_count.item() <= 0:
                return AccumulatedStats()

            stats = AccumulatedStats()
            sample_count = self.acc_count.float().clamp_min(1.0)
            synapse_count = self.synapse_sample_count.clamp_min(1.0)
            dend_count = self.synapse_sample_count.sum(dim=1).clamp_min(1.0)

            stats.x_mean = self.acc_x / synapse_count
            stats.gated_mean = self.acc_gated / dend_count
            stats.post_rate = (self.acc_spikes.float() / sample_count).item()
            stats.total_samples = int(self.acc_count.item())
            stats.spike_count = int(self.acc_spikes.item())
            stats.sparsity = (synapse_count <= 1.0).float().mean().item()

            if self.track_extra:
                stats.v_dend_mean = self.acc_v_dend / sample_count if hasattr(self, "acc_v_dend") else None
                stats.u_mean = (self.acc_u / sample_count).item() if hasattr(self, "acc_u") else None
                stats.theta_mean = (self.acc_theta / sample_count).item() if hasattr(self, "acc_theta") else None
                stats.r_hat_mean = (self.acc_r_hat / sample_count).item() if hasattr(self, "acc_r_hat") else None
                stats.adaptation_mean = (
                    (self.acc_adaptation / sample_count).item() if hasattr(self, "acc_adaptation") else None
                )

            if self.scientific_debug_stats:
                if stats.x_mean is not None:
                    assert torch.isfinite(stats.x_mean).all(), "x_mean contém NaN/Inf"
                if stats.gated_mean is not None:
                    assert torch.isfinite(stats.gated_mean).all(), "gated_mean contém NaN/Inf"

            return stats

    def plot_history(self, keys: Optional[List[str]] = None, figsize=(10, 6)):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib não disponível para plotagem")
            return None

        if not self._history_enabled or not self._history:
            print("Histórico não ativado. Use enable_history(True) primeiro.")
            return None

        if keys is None:
            keys = ["spike_rate", "sparsity", "theta", "r_hat"]

        available_keys = [k for k in keys if k in self._history and self._history[k]]
        if not available_keys:
            print("Nenhum dado histórico disponível")
            return None

        n_plots = len(available_keys)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        for ax, key in zip(axes, available_keys):
            data = self._history[key]
            ax.plot(data)
            ax.set_title(f"{key.replace('_', ' ').title()} over Time")
            ax.set_xlabel("Batch")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @property
    def has_data(self) -> bool:
        return self.initialized.item() and self.acc_count.item() > 0

    @property
    def batch_count(self) -> int:
        return int(self.acc_count.item()) if self.has_data else 0

    def extra_repr(self) -> str:
        status = "active" if self.has_data else "empty"
        history = "on" if self._history_enabled else "off"
        return (
            f"D={self.n_dendrites}, S={self.n_synapses}, mode={self.mode}, "
            f"batches={self.batch_count}, status={status}, track_extra={self.track_extra}, history={history}"
        )


def create_accumulator_from_config(config, track_extra: bool = False) -> StatisticsAccumulator:
    return StatisticsAccumulator(
        n_dendrites=config.n_dendrites,
        n_synapses=config.n_synapses_per_dendrite,
        eps=config.eps,
        track_extra=track_extra,
        mode=getattr(config, "stats_accumulator_mode", "dense"),
        activity_threshold=getattr(config, "activity_threshold", 0.01),
        sparse_min_activity_ratio=getattr(config, "sparse_min_activity_ratio", 0.15),
        scientific_debug_stats=getattr(config, "scientific_debug_stats", False),
        enable_profiling=getattr(config, "enable_accumulator_profiling", False),
    )
