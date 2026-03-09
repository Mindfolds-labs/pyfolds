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
    concept: str
    age_seconds: float
    importance: float
    timestamp: float
    access_count: int = 0
    consolidated: bool = False
    last_access: float = 0.0


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
        self.register_buffer("birth_time", torch.tensor(time.time(), dtype=torch.float32))
        self.register_buffer("age_seconds", torch.tensor(0.0, dtype=torch.float32))

        self._last_meridiem = "AM" if start_phase < 180.0 else "PM"
        self.temporal_memory: Dict[str, List[TemporalMemory]] = {}

    def _advance_circadian(self, dt: float) -> None:
        if not getattr(self, "_circadian_enabled", False):
            return

        delta_phase = (float(dt) / self._circadian_cycle_seconds) * 360.0
        prev_phase = float(self.circadian_phase.item())
        next_phase = (prev_phase + delta_phase) % 360.0
        self.circadian_phase.fill_(next_phase)
        self.age_seconds.add_(float(dt))
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
            self.consolidate_temporal_memory(
                pruning_threshold=float(getattr(self.cfg, "wave_sleep_pruning_threshold", 0.1))
            )
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

    def store_temporal_memory(
        self,
        pattern: torch.Tensor,
        importance: float = 0.5,
        concept: str = "unknown",
    ) -> None:
        if not getattr(self, "_circadian_enabled", False):
            return

        ctx = self._get_circadian_context()
        key = self._circadian_key(float(ctx["phase"]), str(ctx["meridiem"]))
        item = TemporalMemory(
            phase=float(ctx["phase"]),
            context=str(ctx["meridiem"]),
            pattern=pattern.detach().cpu().clone(),
            concept=concept,
            age_seconds=float(self.age_seconds.item()),
            importance=float(importance),
            timestamp=time.time(),
            last_access=time.time(),
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
        for mem in memories[:n]:
            mem.access_count += 1
            mem.last_access = time.time()
        return [m.pattern.to(self.theta.device) for m in memories[:n]]

    def recall_when(self, concept: str) -> List[float]:
        """Retorna idades (em segundos) em que um conceito foi aprendido."""
        if not getattr(self, "_circadian_enabled", False):
            return []

        ages: List[float] = []
        for items in self.temporal_memory.values():
            for memory in items:
                if memory.concept == concept:
                    ages.append(memory.age_seconds)
        return sorted(ages)

    def get_memory_by_age(self, target_age: float, window: float = 3600.0) -> List[TemporalMemory]:
        """Recupera memórias de uma janela ao redor de uma idade alvo."""
        if not getattr(self, "_circadian_enabled", False):
            return []

        result: List[TemporalMemory] = []
        for items in self.temporal_memory.values():
            for memory in items:
                if abs(memory.age_seconds - target_age) <= window:
                    result.append(memory)
        result.sort(key=lambda m: abs(m.age_seconds - target_age))
        return result

    def consolidate_temporal_memory(self, pruning_threshold: float = 0.1) -> Dict[str, int]:
        """Consolida e poda memórias episódicas para uso durante o sono."""
        if not getattr(self, "_circadian_enabled", False):
            return {"consolidated": 0, "pruned": 0}

        consolidated = 0
        pruned = 0
        for key in list(self.temporal_memory.keys()):
            kept: List[TemporalMemory] = []
            for memory in self.temporal_memory[key]:
                if memory.importance < pruning_threshold:
                    pruned += 1
                    continue
                if memory.importance >= 0.7 and not memory.consolidated:
                    memory.consolidated = True
                    consolidated += 1
                kept.append(memory)
            if kept:
                self.temporal_memory[key] = kept
            else:
                del self.temporal_memory[key]

        return {"consolidated": consolidated, "pruned": pruned}

    def narrative_of_life(self) -> str:
        """Gera um resumo curto da história temporal da rede."""
        if not getattr(self, "_circadian_enabled", False):
            return "Temporal memory is disabled."

        age_days = float(self.age_seconds.item()) / (24.0 * 3600.0)
        first_day = self.get_memory_by_age(0.0, window=24.0 * 3600.0)
        recent = self.get_memory_by_age(float(self.age_seconds.item()), window=24.0 * 3600.0)

        all_memories = [m for items in self.temporal_memory.values() for m in items]
        top_memories = sorted(all_memories, key=lambda m: m.importance, reverse=True)[:3]
        top_labels = ", ".join(m.concept for m in top_memories) if top_memories else "none"

        return (
            f"Tenho {age_days:.2f} dias de vida; "
            f"aprendi {len(first_day)} memórias no primeiro dia; "
            f"{len(recent)} nas últimas 24h; "
            f"marcos: {top_labels}."
        )

    def get_temporal_memory_stats(self) -> Dict[str, int]:
        total = sum(len(items) for items in self.temporal_memory.values()) if getattr(self, "_circadian_enabled", False) else 0
        consolidated = (
            sum(1 for items in self.temporal_memory.values() for m in items if m.consolidated)
            if getattr(self, "_circadian_enabled", False)
            else 0
        )
        return {
            "total_memories": total,
            "memory_slots": len(self.temporal_memory) if getattr(self, "_circadian_enabled", False) else 0,
            "consolidated_memories": consolidated,
            "age_seconds": int(self.age_seconds.item()) if getattr(self, "_circadian_enabled", False) else 0,
        }
