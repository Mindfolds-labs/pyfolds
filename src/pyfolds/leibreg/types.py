"""Shared type aliases for LEIBREG."""

from __future__ import annotations

from typing import Dict, TypedDict

import torch


class WordSpaceOutput(TypedDict):
    q_base: torch.Tensor
    q_total: torch.Tensor
    phase_applied: torch.Tensor
    norm: torch.Tensor


TelemetryPayload = Dict[str, float]
