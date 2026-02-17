"""Neurônio MPJRD-Wave: integração cooperativa + codificação por fase."""

from __future__ import annotations

import torch
from typing import Dict, Optional

from ..core.neuron import MPJRDNeuron
from ..utils.types import LearningMode
from ..utils.validation import validate_input
from .config import MPJRDWaveConfig


class MPJRDWaveNeuron(MPJRDNeuron):
    """Neurônio v3.0 com saída oscilatória em quadratura."""

    def __init__(self, cfg: MPJRDWaveConfig, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.cfg: MPJRDWaveConfig = cfg

        self.register_buffer("wave_time", torch.tensor(0.0))
        self.register_buffer("phase_pointer", torch.tensor(0, dtype=torch.long))
        self.register_buffer("phase_history", torch.zeros(cfg.phase_buffer_size))
        self.register_buffer("last_phase_sync", torch.tensor(0.0))

    def _frequency_for_class(self, class_idx: Optional[int]) -> float:
        if self.cfg.class_frequencies is not None:
            if class_idx is None:
                return float(self.cfg.class_frequencies[0])
            idx = int(class_idx) % len(self.cfg.class_frequencies)
            return float(self.cfg.class_frequencies[idx])

        idx = 0 if class_idx is None else int(class_idx)
        return float(self.cfg.base_frequency + idx * self.cfg.frequency_step)

    def _compute_phase(self, u: torch.Tensor) -> torch.Tensor:
        # phase = (pi/2) * (1 - sigmoid(u - theta))
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

    @validate_input(
        expected_ndim=3,
        expected_shape_fn=lambda self: (self.cfg.n_dendrites, self.cfg.n_synapses_per_dendrite),
    )
    def forward(
        self,
        x: torch.Tensor,
        reward: Optional[float] = None,
        mode: Optional[LearningMode] = None,
        collect_stats: bool = True,
        target_class: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        effective_mode = mode if mode is not None else self.mode

        device = self.theta.device
        x = x.to(device)
        B, D, _ = x.shape

        v_dend = torch.zeros(B, D, device=device)
        for d_idx, dend in enumerate(self.dendrites):
            v_dend[:, d_idx] = dend(x[:, d_idx, :])

        dendritic_activations = torch.sigmoid(v_dend - self.cfg.dendritic_threshold)
        u = dendritic_activations.sum(dim=1)
        spikes = (u >= self.theta).float()

        spike_rate = spikes.mean().item()
        saturation_ratio = (self.N == self.cfg.n_max).float().mean().item()

        if effective_mode != LearningMode.INFERENCE and collect_stats:
            self.homeostasis.update(spike_rate)

        if self.cfg.neuromod_mode == "external":
            R_val = float(reward) if reward is not None else 0.0
            R_val = max(-1.0, min(1.0, R_val))
        else:
            R_val = self._compute_R_endogenous(spike_rate, saturation_ratio)

        amplitude = torch.log2(1.0 + torch.relu(u))
        phase = self._compute_phase(u)
        latency = self._compute_latency(amplitude)

        prev_mean_phase = self.phase_history.mean().to(device)
        phase_sync = torch.cos(phase - prev_mean_phase)
        self.last_phase_sync.copy_(phase_sync.mean().detach())

        phase_mod = 1.0 + self.cfg.phase_plasticity_gain * phase_sync.mean().item()
        R_phase = max(-1.0, min(1.0, R_val * phase_mod))
        R_tensor = torch.tensor([R_phase], device=device)

        if collect_stats and effective_mode == LearningMode.BATCH and self.cfg.defer_updates:
            self.stats_acc.accumulate(x.detach(), dendritic_activations.detach(), spikes.detach())

        # telemetria compatível com o pipeline base
        self.step_id.add_(1)
        if self.telemetry is not None and self.telemetry.enabled():
            from ..telemetry import forward_event
            sid = int(self.step_id.item())
            self.telemetry.emit(forward_event(
                step_id=sid,
                mode=self.mode.value,
                spike_rate=spike_rate,
                theta=float(self.theta.item()),
                r_hat=float(self.r_hat.item()),
                saturation_ratio=saturation_ratio,
                R=float(R_tensor.item()),
                N_mean=float(self.N.float().mean().item()),
                I_mean=float(self.I.float().mean().item()),
                W_mean=float(self.W.float().mean().item()),
                phase_mean=float(phase.mean().item()),
                amplitude_mean=float(amplitude.mean().item()),
                frequency_hz=float(self._frequency_for_class(target_class)),
            ))

        # atualiza histórico de fase com média dos exemplos que dispararam
        with torch.no_grad():
            try:
                if spikes.sum() > 0:
                    spike_indices = spikes.bool()
                    if spike_indices.any():
                        phase_mean = phase[spike_indices].mean()
                    else:
                        phase_mean = phase.mean()
                else:
                    phase_mean = phase.mean()
            except Exception as e:
                self.logger.warning(f"Erro ao calcular phase_mean: {e}, usando fallback")
                phase_mean = phase.mean()
            ptr = int(self.phase_pointer.item())
            self.phase_history[ptr] = phase_mean.detach().cpu()
            self.phase_history.mul_(self.cfg.phase_decay)
            self.phase_pointer.copy_(torch.tensor((ptr + 1) % self.cfg.phase_buffer_size))

        self.wave_time.add_(self.cfg.dt)
        frequency_hz = self._frequency_for_class(target_class)

        wave_payload = self._generate_wave_output(
            amplitude=amplitude * spikes,
            phase=phase,
            frequency_hz=frequency_hz,
            t=self.wave_time.to(device),
        )

        arrival_state = torch.where(
            phase < prev_mean_phase,
            torch.ones_like(phase),
            torch.zeros_like(phase),
        )

        return {
            "spikes": spikes,
            "u": u,
            "v_dend": v_dend,
            "gated": dendritic_activations,
            "dendritic_activations": dendritic_activations,
            "theta": self.theta.clone(),
            "r_hat": self.r_hat.clone(),
            "spike_rate": spike_rate,
            "saturation_ratio": saturation_ratio,
            "R": R_tensor,
            "phase": phase,
            "latency": latency,
            "amplitude": amplitude,
            "frequency": torch.tensor(frequency_hz, device=device),
            "phase_sync": phase_sync,
            "arrival_leading": arrival_state,
            **wave_payload,
        }

    @torch.no_grad()
    def apply_plasticity(self, dt: float = 1.0, reward: Optional[float] = None) -> None:
        phase_factor = 1.0 + self.cfg.phase_plasticity_gain * float(self.last_phase_sync.item())
        mod_reward = reward * phase_factor if reward is not None else reward
        super().apply_plasticity(dt=dt, reward=mod_reward)
