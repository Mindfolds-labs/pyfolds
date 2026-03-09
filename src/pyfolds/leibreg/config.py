"""Configuration primitives for the experimental LEIBREG subsystem."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeibregConfig:
    """Configuration for LEIBREG modules.

    Attributes:
        concept_count: Size of learnable concept table.
        dim_base: Core conceptual dimension. Defaults to symbolic 4D.
        dim_context: Optional contextual dimensions appended to the base space.
        normalize_output: Whether to L2-normalize conceptual vectors.
        wave_enabled: Enables optional wave-aware modulation.
        proximity_kernel: Kernel used in proximity attention.
        temperature: Attention temperature; values are clamped in modules for stability.
    """

    concept_count: int = 1024
    dim_base: int = 4
    dim_context: int = 0
    normalize_output: bool = True
    wave_enabled: bool = False
    proximity_kernel: str = "gaussian"
    temperature: float = 1.0
