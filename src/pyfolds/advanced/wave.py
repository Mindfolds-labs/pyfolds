"""Mixin para dinâmica oscilatória (WAVE) com neuromodulação genérica."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..core.config import MPJRDConfig
from ..utils.types import LearningMode, normalize_learning_mode
from .time_mixin import TimedMixin


class WaveNeuromodulator:
    """Neuromodulação genérica alinhada aos modos de execução do PyFolds."""

    def __init__(self, cfg: MPJRDConfig):
        self.cfg = cfg
        self.learning_rate = cfg.wave_learning_rate_gain
        self.focus_gain = cfg.wave_focus_gain
        self.excitation_gain = cfg.wave_excitation_gain
        self.stability_gain = cfg.wave_stability_gain
        self.current_mode = LearningMode.ONLINE

    def set_mode(self, mode: LearningMode) -> None:
        self.current_mode = mode
        if mode == LearningMode.ONLINE:
            self.learning_rate = 1.0
            self.focus_gain = 0.8
            self.excitation_gain = 0.9
            self.stability_gain = 0.3
        elif mode == LearningMode.BATCH:
            self.learning_rate = 0.7
            self.focus_gain = 0.7
            self.excitation_gain = 0.7
            self.stability_gain = 0.7
        elif mode == LearningMode.SLEEP:
            self.learning_rate = 0.2
            self.focus_gain = 0.3
            self.excitation_gain = 0.2
            self.stability_gain = 1.0
        elif mode == LearningMode.INFERENCE:
            self.learning_rate = 0.0
            self.focus_gain = 1.0
            self.excitation_gain = 0.8
            self.stability_gain = 0.8

    def modulate_frequency(self, base_freq: float) -> float:
        return base_freq * (1.0 + 0.5 * self.focus_gain)

    def modulate_amplitude(self, base_amp: torch.Tensor) -> torch.Tensor:
        return base_amp * self.excitation_gain


class WaveOscillator(nn.Module):
    """Oscilador interno do ``WaveMixin``."""

    def __init__(self, frequency: float, trainable: bool = True):
        super().__init__()
        if trainable:
            self.frequency = nn.Parameter(torch.tensor(frequency, dtype=torch.float32))
        else:
            self.register_buffer("frequency", torch.tensor(frequency, dtype=torch.float32))
        self.register_buffer("phase", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("sin_cache", torch.tensor(0.0))
        self.register_buffer("cos_cache", torch.tensor(0.0))

    def update(self, dt: float, neuromod: WaveNeuromodulator) -> Tuple[torch.Tensor, torch.Tensor]:
        freq_eff = neuromod.modulate_frequency(float(self.frequency.item()))
        phase_delta = 2 * math.pi * freq_eff * dt
        self.phase = (self.phase + phase_delta) % (2 * math.pi)
        sin_val = torch.sin(self.phase)
        cos_val = torch.cos(self.phase)
        self.sin_cache = sin_val
        self.cos_cache = cos_val
        return sin_val, cos_val

    def get_wave(self, amplitude: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return amplitude * self.sin_cache, amplitude * self.cos_cache


class WaveMixin(TimedMixin):
    """Mixin de oscilação para composição com ``MPJRDNeuron``."""

    def _init_wave(self, cfg: MPJRDConfig) -> None:
        self._ensure_time_counter()
        self.wave_cfg = cfg
        self.neuromod = WaveNeuromodulator(cfg)

        self.oscillators = nn.ModuleList(
            [
                WaveOscillator(
                    frequency=cfg.wave_base_frequency + i * cfg.wave_frequency_step,
                    trainable=True,
                )
                for i in range(cfg.wave_n_frequencies)
            ]
        )

        self.wave_amplitudes = nn.Parameter(torch.ones(cfg.wave_n_frequencies) * 0.1)
        self.register_buffer("phase_history", torch.zeros(cfg.wave_phase_buffer_size))
        self.register_buffer("phase_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_sync", torch.tensor(0.0))
        self.register_buffer("consolidation_buffer", torch.zeros(1000))

    def set_wave_mode(self, mode: LearningMode | str) -> None:
        mode_val = normalize_learning_mode(mode)
        if mode_val is None:
            return
        self.neuromod.set_mode(mode_val)
        if mode_val == LearningMode.SLEEP and self.wave_cfg.wave_sleep_consolidation:
            self._sleep_consolidation()

    def _update_oscillators(self, dt: float) -> None:
        for osc in self.oscillators:
            osc.update(dt, self.neuromod)

    def _wave_modulate(self, x: torch.Tensor, dt: float = 0.001) -> Tuple[torch.Tensor, Dict]:
        self._update_oscillators(dt)
        combined_sin = torch.zeros(1, device=x.device)

        for i, osc in enumerate(self.oscillators):
            amp = self.neuromod.modulate_amplitude(self.wave_amplitudes[i])
            sin_val, _ = osc.get_wave(amp)
            combined_sin = combined_sin + sin_val

        modulation = combined_sin.abs().mean()
        x_mod = x * (1.0 + 0.1 * modulation)

        wave_state = {
            "wave_modulation": modulation.item(),
            "wave_phase_mean": sum(float(osc.phase.item()) for osc in self.oscillators) / len(self.oscillators),
            "wave_frequencies": [float(osc.frequency.item()) for osc in self.oscillators],
            "wave_amplitudes": self.wave_amplitudes.detach().cpu().tolist(),
            "wave_mode": self.neuromod.current_mode.value,
            "wave_learning_rate": self.neuromod.learning_rate,
            "wave_focus": self.neuromod.focus_gain,
            "wave_excitation": self.neuromod.excitation_gain,
            "wave_stability": self.neuromod.stability_gain,
        }
        return x_mod, wave_state

    def _compute_phase_from_potential(self, u: torch.Tensor) -> torch.Tensor:
        theta = getattr(self, "theta", torch.tensor(0.5, device=u.device))
        if u.dim() == 1:
            delta = (u - theta) * self.wave_cfg.wave_phase_sensitivity
        else:
            delta = (u - theta.unsqueeze(0)) * self.wave_cfg.wave_phase_sensitivity
        phase = (torch.pi / 2.0) * (1.0 - torch.sigmoid(delta))
        return phase.clamp(0.0, torch.pi)

    def _compute_sync(self, u: torch.Tensor) -> torch.Tensor:
        u_phase = self._compute_phase_from_potential(u)
        wave_phase = sum(osc.phase for osc in self.oscillators) / len(self.oscillators)
        return torch.cos(u_phase - wave_phase)

    def _update_sync_memory(self, sync: torch.Tensor) -> None:
        ptr = int(self.phase_ptr.item())
        self.phase_history[ptr] = sync.mean().detach().cpu()
        self.phase_ptr = torch.tensor(
            (ptr + 1) % self.wave_cfg.wave_phase_buffer_size,
            dtype=torch.long,
            device=self.phase_ptr.device,
        )
        self.phase_history.mul_(self.wave_cfg.wave_phase_decay)
        self.last_sync = self.phase_history.mean()

    def _wave_update(self, spikes: torch.Tensor, u: torch.Tensor) -> None:
        if bool((spikes > 0).any()):
            sync = self._compute_sync(u)
            self._update_sync_memory(sync)

    @torch.no_grad()
    def _sleep_consolidation(self) -> None:
        target_norm = 0.1
        current_mean = self.wave_amplitudes.mean()
        if float(current_mean.item()) > 0:
            self.wave_amplitudes.mul_(target_norm / current_mean)
        threshold = self.wave_cfg.wave_sleep_pruning_threshold
        mask = self.wave_amplitudes.abs() > threshold
        self.wave_amplitudes.mul_(mask.float())

    def get_wave_metrics(self) -> Dict[str, float]:
        if not hasattr(self, "oscillators"):
            return {}
        return {
            "wave_n_frequencies": float(len(self.oscillators)),
            "wave_mean_freq": sum(float(osc.frequency.item()) for osc in self.oscillators)
            / len(self.oscillators),
            "wave_mean_amplitude": float(self.wave_amplitudes.mean().item()),
            "wave_last_sync": float(self.last_sync.item()),
            "wave_learning_rate": float(self.neuromod.learning_rate),
            "wave_focus": float(self.neuromod.focus_gain),
            "wave_excitation": float(self.neuromod.excitation_gain),
            "wave_stability": float(self.neuromod.stability_gain),
        }

    def reset_wave_state(self) -> None:
        if not hasattr(self, "oscillators"):
            return
        for osc in self.oscillators:
            osc.phase.zero_()
        self.phase_history.zero_()
        self.phase_ptr.zero_()
        self.last_sync.zero_()
