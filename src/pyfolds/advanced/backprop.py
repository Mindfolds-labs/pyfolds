"""Mixin para backpropagação dendrítica com bAP proporcional."""

import logging
import math
from collections import deque
from typing import Dict, Optional

import torch

from .time_mixin import TimedMixin


logger = logging.getLogger(__name__)


class BackpropMixin(TimedMixin):
    """Mixin para backpropagação dendrítica com opção proporcional."""

    def _backprop_load_state_dict_pre_hook(self, module, state_dict, prefix, *args):
        amp_key = f"{prefix}dendrite_amplification"
        if amp_key in state_dict:
            self.dendrite_amplification.resize_(state_dict[amp_key].shape)
        trace_key = f"{prefix}backprop_trace"
        if trace_key in state_dict:
            self.backprop_trace.resize_(state_dict[trace_key].shape)

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

        if not hasattr(self, "backprop_trace"):
            self.register_buffer("backprop_trace", torch.empty(0))
        if not getattr(self, "_backprop_state_hook_registered", False):
            self.register_load_state_dict_pre_hook(self._backprop_load_state_dict_pre_hook)
            self._backprop_state_hook_registered = True
        self._last_backprop_time = 0.0
        self.backprop_dropped_events = 0

        configured_maxlen = getattr(cfg, "backprop_queue_maxlen", None)
        if configured_maxlen is None:
            max_queue_size = max(100, int(self.backprop_delay * 50))
        else:
            max_queue_size = int(configured_maxlen)
        self.backprop_queue = deque(maxlen=max_queue_size)

    def _ensure_backprop_trace(self, batch_size: int, device: torch.device):
        """Garante que backprop_trace existe com tamanho correto."""
        D = self.cfg.n_dendrites
        S = self.cfg.n_synapses_per_dendrite
        needs_resize = (
            self.backprop_trace.ndim != 3
            or self.backprop_trace.shape[0] != batch_size
            or self.backprop_trace.shape[1] != D
            or self.backprop_trace.shape[2] != S
        )
        if needs_resize:
            self.backprop_trace.resize_(batch_size, D, S)
            self.backprop_trace.zero_()

    def _schedule_backprop(
        self,
        spike_time: float,
        v_dend: torch.Tensor,
        dend_contribution: Optional[torch.Tensor] = None,
    ):
        """Agenda evento de backpropagação."""
        event = {
            "time": spike_time,
            "v_dend": v_dend.detach().clone(),
            "dend_contribution": (
                dend_contribution.detach().clone() if dend_contribution is not None else None
            ),
        }

        if len(self.backprop_queue) == self.backprop_queue.maxlen:
            dropped_event = self.backprop_queue.popleft()
            self.backprop_dropped_events += 1
            local_logger = getattr(self, "logger", logger)
            local_logger.warning(
                "event=backprop_queue_overflow capacity=%d dropped_event_time=%.6f incoming_event_time=%.6f",
                self.backprop_queue.maxlen,
                dropped_event["time"],
                event["time"],
            )
        self.backprop_queue.append(event)

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

        if self.backprop_trace.numel() > 0:
            self.backprop_trace.mul_(decay_trace)

        self._last_backprop_time = current_time

        while self.backprop_queue and current_time >= self.backprop_queue[0]["time"]:
            event = self.backprop_queue.popleft()
            v_dend = event["v_dend"]
            dend_contribution = event["dend_contribution"]

            batch_size = v_dend.shape[0]
            device = v_dend.device
            self._ensure_backprop_trace(batch_size, device)

            uses_proportional = self.bap_proportional and dend_contribution is not None
            source = dend_contribution if uses_proportional else v_dend

            if uses_proportional:
                base_gain = source
            else:
                base_gain = torch.sigmoid(source / 5.0)

            amplification_gain = self.backprop_signal * base_gain.mean(dim=0)
            self.dendrite_amplification.add_(
                amplification_gain.clamp(max=self.backprop_max_amp)
            )

            active_mask = source > self.backprop_active_threshold
            self.backprop_trace.add_(
                active_mask.unsqueeze(-1).to(self.backprop_trace.dtype) * self.backprop_signal
            )
            self.backprop_trace.clamp_(max=2.0)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass com backpropagação."""
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
        if self.backprop_trace.numel() > 0:
            output["backprop_trace_mean"] = self.backprop_trace.mean()

        return output

    def reset_backprop(self):
        """Reseta estado de backpropagação."""
        self.dendrite_amplification.zero_()
        self.backprop_trace.resize_(0)
        self.backprop_queue.clear()
        self._last_backprop_time = 0.0
        self.backprop_dropped_events = 0

    def get_backprop_metrics(self) -> dict:
        """Retorna métricas de backpropagação."""
        metrics = {
            "backprop_amp_mean": self.dendrite_amplification.mean().item(),
            "backprop_amp_max": self.dendrite_amplification.max().item(),
            "backprop_queue_len": len(self.backprop_queue),
            "backprop_dropped_events": self.backprop_dropped_events,
            "bap_proportional": self.bap_proportional,
        }
        if self.backprop_trace.numel() > 0:
            metrics["backprop_trace_mean"] = self.backprop_trace.mean().item()
        return metrics
