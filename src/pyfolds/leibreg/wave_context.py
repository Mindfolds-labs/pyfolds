"""Optional adapters around existing wave state for LEIBREG.

This adapter intentionally stays lightweight to avoid coupling with stable core
internals while still reusing the existing WaveMixin interface when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class WaveContext:
    """Wave context extracted from a source object."""

    phase_mean: Optional[float]


def extract_wave_phase(source: Any) -> Optional[torch.Tensor]:
    """Extract a scalar phase tensor from an object exposing ``oscillators``.

    Returns ``None`` if source does not look wave-compatible.
    """
    oscillators = getattr(source, "oscillators", None)
    if oscillators is None:
        return None

    phases = []
    for osc in oscillators:
        phase = getattr(osc, "phase", None)
        if phase is not None:
            phases.append(float(phase.item()))
    if not phases:
        return None
    return torch.tensor(sum(phases) / len(phases), dtype=torch.float32)
