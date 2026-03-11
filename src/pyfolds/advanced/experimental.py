"""Infraestrutura para mecanismos experimentais com toggles controlados."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class MechanismSpec:
    """Contrato mínimo para mecanismo opcional."""

    name: str
    flag: str
    application_point: str


@dataclass(frozen=True)
class ExperimentalMechanismConfig:
    """Snapshot explícito dos toggles experimentais suportados."""

    enable_phase_gating: bool = False
    enable_dynamic_channel_gating: bool = False
    enable_wave_modulation: bool = False
    enable_sleep_consolidation: bool = True
    enable_dendritic_threshold_modulation: bool = False
    debug_compare_baseline: bool = False
    debug_collect_mechanism_metrics: bool = False

    @classmethod
    def from_config(cls, cfg: Any) -> "ExperimentalMechanismConfig":
        return cls(
            enable_phase_gating=bool(cfg.enable_phase_gating),
            enable_dynamic_channel_gating=bool(cfg.enable_dynamic_channel_gating),
            enable_wave_modulation=bool(cfg.enable_wave_modulation),
            enable_sleep_consolidation=bool(cfg.enable_sleep_consolidation),
            enable_dendritic_threshold_modulation=bool(cfg.enable_dendritic_threshold_modulation),
            debug_compare_baseline=bool(cfg.debug_compare_baseline),
            debug_collect_mechanism_metrics=bool(cfg.debug_collect_mechanism_metrics),
        )


class MechanismToggleSet:
    """Registry leve para evitar condicionais dispersas."""

    SPECS = {
        "phase_gating": MechanismSpec("phase_gating", "enable_phase_gating", "advanced.stdp:update_delta_w"),
        "dynamic_channel_gating": MechanismSpec(
            "dynamic_channel_gating",
            "enable_dynamic_channel_gating",
            "advanced.stdp:update_delta_w",
        ),
        "wave_modulation": MechanismSpec("wave_modulation", "enable_wave_modulation", "advanced.wave:forward"),
        "sleep_consolidation": MechanismSpec(
            "sleep_consolidation", "enable_sleep_consolidation", "core.synapse:consolidate"
        ),
        "dendritic_threshold_modulation": MechanismSpec(
            "dendritic_threshold_modulation",
            "enable_dendritic_threshold_modulation",
            "core.neuron:dendritic_threshold",
        ),
    }

    def __init__(self, cfg: Any):
        self._cfg = cfg

    def is_enabled(self, mechanism_name: str) -> bool:
        spec = self.SPECS.get(mechanism_name)
        if spec is None:
            raise KeyError(f"Unknown mechanism '{mechanism_name}'.")
        return bool(getattr(self._cfg, spec.flag))

    def all_status(self) -> dict[str, bool]:
        return {name: self.is_enabled(name) for name in self.SPECS}


@dataclass(frozen=True)
class MechanismComparisonReport:
    output_diff: dict[str, float]
    baseline_metrics: dict[str, float]
    experiment_metrics: dict[str, float]


def diff_output_stats(baseline_output: dict[str, Any], experiment_output: dict[str, Any]) -> dict[str, float]:
    """Calcula diferenças absolutas médias para tensores comuns nas saídas."""
    diffs: dict[str, float] = {}
    for key in sorted(set(baseline_output.keys()) & set(experiment_output.keys())):
        a = baseline_output[key]
        b = experiment_output[key]
        if torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape:
            if a.dtype == torch.bool or b.dtype == torch.bool:
                diffs[f"{key}_mean_abs_diff"] = float((a.float() - b.float()).abs().mean().item())
            else:
                diffs[f"{key}_mean_abs_diff"] = float((a - b).abs().mean().item())
    return diffs


def collect_mechanism_report(output: dict[str, Any]) -> dict[str, float]:
    """Extrai métricas objetivas para comparação A/B sem alterar o forward."""
    spikes = output.get("spikes")
    stats = output.get("stats")

    report: dict[str, float] = {
        "spike_rate": float(spikes.float().mean().item()) if torch.is_tensor(spikes) else 0.0,
        "average_weight_update": float(output.get("average_weight_update", 0.0)),
        "dendritic_activity_rate": 0.0,
        "refractory_block_rate": float(output.get("refractory_block_rate", 0.0)),
        "adaptation_level_mean": float(output.get("adaptation_level_mean", 0.0)),
        "phase_alignment_mean": float(output.get("phase_alignment_mean", 0.0)),
        "sparsity_ratio": 0.0,
        "active_dendrite_ratio": float(output.get("active_dendrite_ratio", 0.0)),
        "learning_event_count": float(output.get("learning_event_count", 0.0)),
    }

    if torch.is_tensor(stats):
        report["dendritic_activity_rate"] = float((stats > 0).float().mean().item())
    if torch.is_tensor(spikes):
        report["sparsity_ratio"] = float((spikes == 0).float().mean().item())
    return report


def compare_mechanism_vs_baseline(
    *,
    factory: Callable[[bool], Any],
    x: torch.Tensor,
    mechanism_name: str,
    forward_kwargs: dict[str, Any] | None = None,
) -> MechanismComparisonReport:
    """Roda A/B reproduzível: baseline vs mecanismo ligado, com estado inicial idêntico."""
    kwargs = dict(forward_kwargs or {})
    baseline_model = factory(False)
    experiment_model = factory(True)
    experiment_model.load_state_dict(baseline_model.state_dict())

    with torch.no_grad():
        baseline_output = baseline_model(x, **kwargs)
        experiment_output = experiment_model(x, **kwargs)

    baseline_metrics = collect_mechanism_report(baseline_output)
    experiment_metrics = collect_mechanism_report(experiment_output)
    output_diff = diff_output_stats(baseline_output, experiment_output)
    output_diff["mechanism_enabled"] = float(
        MechanismToggleSet(getattr(experiment_model, "cfg", object())).is_enabled(mechanism_name)
    )

    return MechanismComparisonReport(
        output_diff=output_diff,
        baseline_metrics=baseline_metrics,
        experiment_metrics=experiment_metrics,
    )
