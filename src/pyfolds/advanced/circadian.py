"""Circadian modulation mixin for wave-enabled advanced neurons."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List

import torch

from ..utils.types import LearningMode


@dataclass
class TemporalMemory:
    """Stored memory indexed by circadian phase and meridiem."""

    phase: float
    context: str
    pattern: torch.Tensor
    importance: float
    timestamp: float


class CircadianWaveMixin:
    """Adds 12h AM/PM time awareness and temporal memory to wave outputs.

    When enabled, this mixin can also drive learning-mode transitions:
    - AM -> ONLINE (active learning)
    - PM -> SLEEP (consolidation)
    """

    def _init_circadian(self, cfg) -> None:
        self._circadian_enabled = bool(getattr(cfg, "circadian_enabled", False))
        if not self._circadian_enabled:
            return

        self._circadian_cycle_seconds = float(cfg.circadian_cycle_hours) * 3600.0
        self._circadian_phase_bins = int(cfg.circadian_phase_bins)
        self._circadian_auto_mode = bool(getattr(cfg, "circadian_auto_mode", False))
        self._circadian_sleep_duration = float(getattr(cfg, "circadian_sleep_duration", 60.0))

        self._circadian_am_cortisol = float(cfg.circadian_am_cortisol)
        self._circadian_pm_cortisol = float(cfg.circadian_pm_cortisol)
        self._circadian_am_melatonin = float(cfg.circadian_am_melatonin)
        self._circadian_pm_melatonin = float(cfg.circadian_pm_melatonin)

        start_hour = float(cfg.circadian_day_start_hour)
        start_phase = ((start_hour % float(cfg.circadian_cycle_hours)) / float(cfg.circadian_cycle_hours)) * 360.0
        self.register_buffer("circadian_phase", torch.tensor(start_phase))
        self.register_buffer("circadian_day", torch.tensor(0, dtype=torch.long))

        self._last_meridiem = "AM" if start_phase < 180.0 else "PM"
        self.temporal_memory: Dict[str, List[TemporalMemory]] = {}

    def _advance_circadian(self, dt: float) -> None:
        if not getattr(self, "_circadian_enabled", False):
            return

        delta_phase = (float(dt) / self._circadian_cycle_seconds) * 360.0
        prev_phase = float(self.circadian_phase.item())
        next_phase = (prev_phase + delta_phase) % 360.0
        self.circadian_phase.fill_(next_phase)
        if next_phase < prev_phase:
            self.circadian_day.add_(1)

    def _circadian_key(self, phase: float, meridiem: str) -> str:
        phase_bin = int((phase / 360.0) * self._circadian_phase_bins) % self._circadian_phase_bins
        return f"{phase_bin}_{meridiem}"

    def _get_circadian_context(self) -> Dict[str, float | str]:
        phase = float(self.circadian_phase.item())
        meridiem = "AM" if phase < 180.0 else "PM"

        if meridiem == "AM":
            cortisol = self._circadian_am_cortisol
            melatonin = self._circadian_am_melatonin
        else:
            cortisol = self._circadian_pm_cortisol
            melatonin = self._circadian_pm_melatonin

        hour = (phase / 360.0) * float(self.cfg.circadian_cycle_hours)
        focus_gain = cortisol * (1.0 - 0.5 * melatonin)
        return {
            "phase": phase,
            "hour": hour,
            "meridiem": meridiem,
            "day": int(self.circadian_day.item()),
            "cortisol": cortisol,
            "melatonin": melatonin,
            "focus_gain": focus_gain,
            "recommended_mode": LearningMode.ONLINE if meridiem == "AM" else LearningMode.SLEEP,
        }

    def _apply_circadian_mode(self, ctx: Dict[str, float | str]) -> None:
        """Synchronize neuron mode with circadian context (optional)."""
        if not self._circadian_auto_mode:
            self._last_meridiem = str(ctx["meridiem"])
            return

        target_mode = ctx["recommended_mode"]
        if target_mode != self.mode:
            self.set_mode(target_mode)

        meridiem = str(ctx["meridiem"])
        entered_pm = self._last_meridiem != meridiem and meridiem == "PM"
        if entered_pm:
            # Trigger consolidation once on AM->PM transition.
            self.sleep(duration=self._circadian_sleep_duration)
        self._last_meridiem = meridiem

    def _get_circadian_embedding(self, ctx: Dict[str, float | str], device: torch.device) -> torch.Tensor:
        phase_rad = math.radians(float(ctx["phase"]))
        return torch.tensor(
            [
                math.sin(phase_rad),
                math.cos(phase_rad),
                math.sin(2.0 * phase_rad),
                math.cos(2.0 * phase_rad),
                1.0 if ctx["meridiem"] == "AM" else 0.0,
                1.0 if ctx["meridiem"] == "PM" else 0.0,
                float(ctx["day"]) / 100.0,
            ],
            device=device,
            dtype=torch.float32,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        output = super().forward(x, **kwargs)
        if not getattr(self, "_circadian_enabled", False):
            return output

        dt = kwargs.get("dt", 1.0)
        self._advance_circadian(dt=dt)
        ctx = self._get_circadian_context()
        self._apply_circadian_mode(ctx)

        if "frequency" in output:
            output["frequency"] = output["frequency"] * float(1.0 + 0.1 * ctx["focus_gain"])

        output.update(
            {
                "circadian_phase": torch.tensor(ctx["phase"], device=output["u"].device),
                "circadian_hour": torch.tensor(ctx["hour"], device=output["u"].device),
                "circadian_meridiem": ctx["meridiem"],
                "circadian_day": torch.tensor(ctx["day"], device=output["u"].device),
                "circadian_embedding": self._get_circadian_embedding(ctx, device=output["u"].device),
                "neuromod_cortisol": torch.tensor(ctx["cortisol"], device=output["u"].device),
                "neuromod_melatonin": torch.tensor(ctx["melatonin"], device=output["u"].device),
                "circadian_mode": self.mode.value,
            }
        )
        return output

    def store_temporal_memory(self, pattern: torch.Tensor, importance: float = 0.5) -> None:
        if not getattr(self, "_circadian_enabled", False):
            return

        ctx = self._get_circadian_context()
        key = self._circadian_key(float(ctx["phase"]), str(ctx["meridiem"]))
        item = TemporalMemory(
            phase=float(ctx["phase"]),
            context=str(ctx["meridiem"]),
            pattern=pattern.detach().cpu().clone(),
            importance=float(importance),
            timestamp=time.time(),
        )

        bucket = self.temporal_memory.setdefault(key, [])
        bucket.append(item)
        bucket.sort(key=lambda m: m.importance, reverse=True)
        self.temporal_memory[key] = bucket[:10]

    def recall_temporal_memories(self, n: int = 5) -> List[torch.Tensor]:
        if not getattr(self, "_circadian_enabled", False):
            return []

        ctx = self._get_circadian_context()
        key = self._circadian_key(float(ctx["phase"]), str(ctx["meridiem"]))
        memories = self.temporal_memory.get(key, [])
        return [m.pattern.to(self.theta.device) for m in memories[:n]]

    def get_temporal_memory_stats(self) -> Dict[str, int]:
        total = sum(len(items) for items in self.temporal_memory.values()) if getattr(self, "_circadian_enabled", False) else 0
        return {
            "total_memories": total,
            "memory_slots": len(self.temporal_memory) if getattr(self, "_circadian_enabled", False) else 0,
        }
