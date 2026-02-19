"""Mixin para backpropagação dendrítica com bAP proporcional."""

import math
from collections import deque
from typing import Dict, Optional

import torch

from .time_mixin import TimedMixin


class BackpropMixin(TimedMixin):
    """Mixin para backpropagação dendrítica com opção proporcional."""

    def _init_backprop(self, cfg):
        """Inicializa parâmetros de backpropagação a partir da config."""
        self.backprop_delay = cfg.backprop_delay
        self.backprop_signal = cfg.backprop_signal
        self.backprop_amp_tau = cfg.backprop_amp_tau
        self.backprop_trace_tau = cfg.backprop_trace_tau
        self.backprop_max_amp = cfg.backprop_max_amp
        self.backprop_max_gain = cfg.backprop_max_gain
        self.backprop_active_threshold = getattr(cfg, "backprop_active_threshold", 0.1)
        self.bap_proportional = getattr(cfg, "bap_proportional", True)

        self._ensure_time_counter()

        D = self.cfg.n_dendrites
        self.register_buffer("dendrite_amplification", torch.zeros(D))

        self.backprop_trace = None
        self._last_backprop_time = 0.0

        max_queue_size = max(100, int(self.backprop_delay * 50))
        self.backprop_queue = deque(maxlen=max_queue_size)

    def _ensure_backprop_trace(self, batch_size: int, device: torch.device):
        """Garante que backprop_trace existe com tamanho correto."""
        D = self.cfg.n_dendrites
        S = self.cfg.n_synapses_per_dendrite
        if self.backprop_trace is None or self.backprop_trace.shape[0] != batch_size:
            self.backprop_trace = torch.zeros(batch_size, D, S, device=device)

    def _schedule_backprop(
        self,
        spike_time: float,
        v_dend: torch.Tensor,
        dend_contribution: Optional[torch.Tensor] = None,
    ):
        """Agenda evento de backpropagação."""
        self.backprop_queue.append(
            {
                "time": spike_time,
                "v_dend": v_dend.detach().clone(),
                "dend_contribution": (
                    dend_contribution.detach().clone()
                    if dend_contribution is not None
                    else None
                ),
            }
        )

    def _process_backprop_queue(self, current_time: float):
        """Processa eventos de backpropagação pendentes."""
        if not self.backprop_queue:
            return

        time_since_last = max(0.0, current_time - self._last_backprop_time)
        decay_amp = math.exp(-time_since_last / self.backprop_amp_tau)
        decay_trace = math.exp(-time_since_last / self.backprop_trace_tau)

        self.dendrite_amplification.mul_(decay_amp)

        first_event = self.backprop_queue[0]
        batch_size = first_event["v_dend"].shape[0]
        device = first_event["v_dend"].device
        self._ensure_backprop_trace(batch_size, device)

        if self.backprop_trace is not None:
            self.backprop_trace.mul_(decay_trace)

        self._last_backprop_time = current_time

        while self.backprop_queue and current_time >= self.backprop_queue[0]["time"]:
            event = self.backprop_queue.popleft()
            v_dend = event["v_dend"]
            dend_contribution = event["dend_contribution"]

            batch_size = v_dend.shape[0]
            device = v_dend.device
            self._ensure_backprop_trace(batch_size, device)

            if self.bap_proportional and dend_contribution is not None:
                amplification_gain = self.backprop_signal * dend_contribution.mean(dim=0)
            else:
                activity_factor = torch.sigmoid(v_dend / 5.0)
                amplification_gain = self.backprop_signal * activity_factor.mean(dim=0)

            self.dendrite_amplification.add_(
                amplification_gain.clamp(max=self.backprop_max_amp)
            )

            for d_idx in range(self.cfg.n_dendrites):
                if self.bap_proportional and dend_contribution is not None:
                    active_samples = (
                        dend_contribution[:, d_idx] > self.backprop_active_threshold
                    )
                else:
                    active_samples = (
                        v_dend[:, d_idx] > self.backprop_active_threshold
                    )

                self.backprop_trace[active_samples, d_idx, :] += self.backprop_signal

            self.backprop_trace.clamp_(max=2.0)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass com backpropagação."""
        dt = kwargs.get("dt", 1.0)
        current_time = self.time_counter.item()

        current_time = self.time_counter.item()
        self._process_backprop_queue(current_time)

        output = super().forward(x, **kwargs)

        if output["spikes"].any():
            self._schedule_backprop(
                spike_time=current_time + self.backprop_delay,
                v_dend=output["v_dend"],
                dend_contribution=output.get("dend_contribution"),
            )

        output["dendrite_amplification"] = self.dendrite_amplification.clone()
        if self.backprop_trace is not None:
            output["backprop_trace_mean"] = self.backprop_trace.mean()

        self._increment_time(dt)

        return output

    def reset_backprop(self):
        """Reseta estado de backpropagação."""
        self.dendrite_amplification.zero_()
        self.backprop_trace = None
        self.backprop_queue.clear()
        self._last_backprop_time = 0.0

    def get_backprop_metrics(self) -> dict:
        """Retorna métricas de backpropagação."""
        metrics = {
            "backprop_amp_mean": self.dendrite_amplification.mean().item(),
            "backprop_amp_max": self.dendrite_amplification.max().item(),
            "backprop_queue_len": len(self.backprop_queue),
            "bap_proportional": self.bap_proportional,
        }
        if self.backprop_trace is not None:
            metrics["backprop_trace_mean"] = self.backprop_trace.mean().item()
        return metrics
