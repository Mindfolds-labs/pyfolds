"""Mixin para dinâmica wave opcional em neurônios avançados."""

from __future__ import annotations

from typing import Dict, Optional

import torch


class WaveDynamicsMixin:
    """Adiciona codificação por fase/frequência de forma opcional.

    A ativação é controlada por ``cfg.wave_enabled`` para evitar a necessidade
    de separar NeuroConfig/WaveConfig no uso cotidiano.
    """

    def _init_wave_dynamics(self, cfg) -> None:
        self._wave_enabled = bool(getattr(cfg, "wave_enabled", False))
        if not self._wave_enabled:
            return

        self.register_buffer("wave_time", torch.tensor(0.0))
        self.register_buffer("phase_pointer", torch.tensor(0, dtype=torch.long))
        self.register_buffer("phase_history", torch.zeros(int(cfg.phase_buffer_size)))
        self.register_buffer("last_phase_sync", torch.tensor(0.0))

    def _frequency_for_class(self, class_idx: Optional[int]) -> float:
        freqs = getattr(self.cfg, "class_frequencies", None)
        if freqs is not None:
            if class_idx is None:
                return float(freqs[0])
            return float(freqs[int(class_idx) % len(freqs)])

        idx = 0 if class_idx is None else int(class_idx)
        return float(self.cfg.base_frequency + idx * self.cfg.frequency_step)

    def _compute_phase(self, u: torch.Tensor) -> torch.Tensor:
        delta = (u - self.theta) * self.cfg.phase_sensitivity
        phase = (torch.pi / 2.0) * (1.0 - torch.sigmoid(delta))
        return phase.clamp(0.0, torch.pi)

    def _compute_latency(self, amplitude: torch.Tensor) -> torch.Tensor:
        return self.cfg.latency_scale / (amplitude + self.cfg.amplitude_eps)

    def _generate_wave_output(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
        frequency_hz: float,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        omega_t = 2.0 * torch.pi * frequency_hz * t
        angle = omega_t + phase
        return {
            "wave_real": amplitude * torch.cos(angle),
            "wave_imag": amplitude * torch.sin(angle),
            "wave_complex": torch.complex(amplitude * torch.cos(angle), amplitude * torch.sin(angle)),
        }

    def forward(self, x: torch.Tensor, **kwargs):
        output = super().forward(x, **kwargs)

        if not getattr(self, "_wave_enabled", False):
            return output

        device = output["u"].device
        dt = kwargs.get("dt", 1.0)
        target_class = kwargs.get("target_class")

        u = output["u"]
        spikes = output["spikes"]
        amplitude = torch.log2(1.0 + torch.relu(u))
        phase = self._compute_phase(u)
        latency = self._compute_latency(amplitude)

        prev_mean_phase = self.phase_history.mean().to(device)
        phase_sync = torch.cos(phase - prev_mean_phase)
        self.last_phase_sync.copy_(phase_sync.mean().detach())

        with torch.no_grad():
            phase_mean = phase[spikes.bool()].mean() if spikes.any() else phase.mean()
            ptr = int(self.phase_pointer.item())
            self.phase_history[ptr] = phase_mean.detach().cpu()
            self.phase_history.mul_(self.cfg.phase_decay)
            self.phase_pointer.copy_(torch.tensor((ptr + 1) % self.cfg.phase_buffer_size))

        self.wave_time.add_(dt)
        frequency_hz = self._frequency_for_class(target_class)
        wave_payload = self._generate_wave_output(
            amplitude=amplitude * spikes,
            phase=phase,
            frequency_hz=frequency_hz,
            t=self.wave_time.to(device),
        )

        output.update(
            {
                "phase": phase,
                "latency": latency,
                "amplitude": amplitude,
                "frequency": torch.tensor(frequency_hz, device=device),
                "phase_sync": phase_sync,
                **wave_payload,
            }
        )
        return output

    @torch.no_grad()
    def apply_plasticity(self, dt: float = 1.0, reward: Optional[float] = None) -> None:
        if getattr(self, "_wave_enabled", False) and reward is not None:
            phase_factor = 1.0 + self.cfg.phase_plasticity_gain * float(self.last_phase_sync.item())
            reward = reward * phase_factor
        super().apply_plasticity(dt=dt, reward=reward)
