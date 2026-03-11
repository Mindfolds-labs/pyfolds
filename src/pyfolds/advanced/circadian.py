"""Circadian modulation mixin for wave-enabled advanced neurons."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Protocol, runtime_checkable

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


@runtime_checkable
class _CircadianHost(Protocol):
    cfg: object
    mode: LearningMode
    theta: torch.Tensor

    def set_mode(self, mode: LearningMode) -> None: ...
    def sleep(self, duration: float = 60.0) -> None: ...
    def queue_runtime_injection(self, name: str, value: object) -> None: ...


class CircadianWaveMixin:
    """Adds 12h AM/PM time awareness and temporal memory to wave outputs.

    When enabled, this mixin can also drive learning-mode transitions:
    - AM -> ONLINE (active learning)
    - PM -> SLEEP (consolidation)
    """

    def _init_circadian(self, cfg) -> None:
        if not isinstance(self, _CircadianHost):
            raise TypeError(
                "CircadianWaveMixin requer host com contrato explícito de modo/sono/config"
            )
        self._circadian_enabled = bool(getattr(cfg, "circadian_enabled", False)) and bool(getattr(cfg, "experimental_circadian_enabled", True))
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
        self._enable_sleep_consolidation = bool(
            getattr(cfg, "enable_sleep_consolidation", getattr(cfg, "wave_sleep_consolidation", True))
        )
        self._offline_pipeline_state: Dict[str, int | str] = {
            "consolidation_requested": 0,
            "consolidation_executed": 0,
            "consolidation_skipped_disabled": 0,
            "replay_runs": 0,
            "last_consolidated": 0,
            "last_pruned": 0,
            "last_trigger": "init",
        }

    def _advance_circadian(self, dt: float) -> None:
        if not getattr(self, "_circadian_enabled", False):
            return

        delta_phase = (float(dt) / self._circadian_cycle_seconds) * 360.0
        prev_phase = float(self.circadian_phase.item())
        next_phase = (prev_phase + delta_phase) % 360.0
        self.circadian_phase.fill_(next_phase)
        self.age_seconds.add_(float(dt))
        if hasattr(self, "time_counter"):
            self.time_counter.fill_(float(self.age_seconds.item()))
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
            # Trigger sleep + offline consolidation once on AM->PM transition.
            self.sleep(duration=self._circadian_sleep_duration)
            self.consolidate_memories(trigger="circadian_pm_transition")
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

    def _apply_circadian_plasticity_gate(self, ctx: Dict[str, float | str]) -> float:
        """Calcula multiplicador de plasticidade baseado na fase circadiana.

        Parameters
        ----------
        ctx : Dict[str, float | str]
            Contexto circadiano atual.

        Returns
        -------
        float
            Multiplicador contínuo entre os limites configurados.
        """
        phase_rad = math.radians(float(ctx["phase"]))
        circadian_gate = 0.5 + 0.5 * math.cos(phase_rad)
        min_gate = float(getattr(self.cfg, "circadian_plasticity_min", 0.1))
        max_gate = float(getattr(self.cfg, "circadian_plasticity_max", 1.5))
        return min_gate + (max_gate - min_gate) * circadian_gate

    def forward(self, x: torch.Tensor, **kwargs):
        phase_snapshot = float(self.circadian_phase.item()) if getattr(self, "_circadian_enabled", False) else 0.0
        output = super().forward(x, **kwargs)
        if not getattr(self, "_circadian_enabled", False):
            return output

        # O core também possui estado circadiano próprio; restauramos o snapshot
        # para manter este mixin como fonte de verdade do ciclo offline/temporal.
        self.circadian_phase.fill_(phase_snapshot)
        dt = kwargs.get("dt", 1.0)
        self._advance_circadian(dt=dt)
        ctx = self._get_circadian_context()
        self._apply_circadian_mode(ctx)

        if "frequency" in output:
            output["frequency"] = output["frequency"] * float(1.0 + 0.1 * ctx["focus_gain"])

        plasticity_gate = self._apply_circadian_plasticity_gate(ctx)
        if hasattr(self, "queue_runtime_injection"):
            effective_eta = float(self.cfg.i_eta) * plasticity_gate
            self.queue_runtime_injection("i_eta", effective_eta)

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
                "circadian_plasticity_gate": plasticity_gate,
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

    def consolidate_temporal_memory(
        self,
        pruning_threshold: float = 0.1,
        decay_rate: float = 0.95,
        access_boost: float = 0.1,
    ) -> Dict[str, int]:
        """Consolida memórias com curva de esquecimento de Ebbinghaus.

        Parameters
        ----------
        pruning_threshold : float, default=0.1
            Importância mínima para manter a memória após decaimento.
        decay_rate : float, default=0.95
            Fator de decaimento temporal por ciclo de consolidação.
        access_boost : float, default=0.1
            Reforço multiplicativo baseado em contagem de acessos.

        Returns
        -------
        Dict[str, int]
            Contagem de memórias consolidadas e podadas.
        """
        if not getattr(self, "_circadian_enabled", False):
            return {"consolidated": 0, "pruned": 0}

        consolidated, pruned = 0, 0
        current_age = float(self.age_seconds.item())

        for key in list(self.temporal_memory.keys()):
            kept: List[TemporalMemory] = []
            for memory in self.temporal_memory[key]:
                age_delta = max(0.0, current_age - float(memory.age_seconds))
                age_factor = math.exp(-decay_rate * age_delta / 3600.0)
                access_factor = 1.0 + access_boost * float(memory.access_count)
                effective_importance = float(memory.importance) * age_factor * access_factor
                memory.importance = min(1.0, effective_importance)

                if memory.importance < pruning_threshold:
                    pruned += 1
                    continue

                if memory.importance >= 0.7 and not memory.consolidated:
                    memory.consolidated = True
                    consolidated += 1

                kept.append(memory)

            if kept:
                kept.sort(key=lambda m: m.importance, reverse=True)
                self.temporal_memory[key] = kept[:10]
            else:
                del self.temporal_memory[key]

        return {"consolidated": consolidated, "pruned": pruned}


    def consolidate_memories(
        self,
        trigger: str = "manual",
        include_replay: bool = False,
        pruning_threshold: float | None = None,
    ) -> Dict[str, int | bool]:
        if not getattr(self, "_circadian_enabled", False):
            return {"executed": False, "replay": False, "consolidated": 0, "pruned": 0}

        self._offline_pipeline_state["consolidation_requested"] = int(
            self._offline_pipeline_state["consolidation_requested"]
        ) + 1
        self._offline_pipeline_state["last_trigger"] = trigger

        if not self._enable_sleep_consolidation:
            self._offline_pipeline_state["consolidation_skipped_disabled"] = int(
                self._offline_pipeline_state["consolidation_skipped_disabled"]
            ) + 1
            return {"executed": False, "replay": False, "consolidated": 0, "pruned": 0}

        replay_done = False
        if include_replay and hasattr(self, "run_replay_cycle"):
            self.run_replay_cycle()
            replay_done = True
            self._offline_pipeline_state["replay_runs"] = int(self._offline_pipeline_state["replay_runs"]) + 1

        report = self.consolidate_temporal_memory(
            pruning_threshold=float(
                pruning_threshold
                if pruning_threshold is not None
                else getattr(self.cfg, "wave_sleep_pruning_threshold", 0.1)
            )
        )
        self._offline_pipeline_state["consolidation_executed"] = int(
            self._offline_pipeline_state["consolidation_executed"]
        ) + 1
        self._offline_pipeline_state["last_consolidated"] = int(report["consolidated"])
        self._offline_pipeline_state["last_pruned"] = int(report["pruned"])
        return {
            "executed": True,
            "replay": replay_done,
            "consolidated": int(report["consolidated"]),
            "pruned": int(report["pruned"]),
        }

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
        stats = {
            "total_memories": total,
            "memory_slots": len(self.temporal_memory) if getattr(self, "_circadian_enabled", False) else 0,
            "consolidated_memories": consolidated,
            "age_seconds": int(self.age_seconds.item()) if getattr(self, "_circadian_enabled", False) else 0,
        }
        if getattr(self, "_circadian_enabled", False):
            stats.update(
                {
                    "sleep_consolidation_enabled": int(self._enable_sleep_consolidation),
                    "offline_consolidation_requested": int(self._offline_pipeline_state["consolidation_requested"]),
                    "offline_consolidation_executed": int(self._offline_pipeline_state["consolidation_executed"]),
                    "offline_consolidation_skipped": int(self._offline_pipeline_state["consolidation_skipped_disabled"]),
                    "offline_replay_runs": int(self._offline_pipeline_state["replay_runs"]),
                    "offline_last_consolidated": int(self._offline_pipeline_state["last_consolidated"]),
                    "offline_last_pruned": int(self._offline_pipeline_state["last_pruned"]),
                }
            )
        return stats
