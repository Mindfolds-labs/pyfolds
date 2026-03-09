"""Configuration primitives for LEIBREG."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeibregConfig:
    concept_count: int = 1024
    dim_base: int = 4
    dim_context: int = 0
    normalize_output: bool = True
    wave_enabled: bool = True
    reg_depth: int = 2
    init_temperature: float = 0.1
    sigreg_enabled: bool = False
    sigreg_weight: float = 0.1
