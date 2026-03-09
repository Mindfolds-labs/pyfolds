"""Shared type aliases for LEIBREG."""

from __future__ import annotations

from typing import Dict, Optional, TypedDict

import torch


class WordSpaceOutput(TypedDict):
    """Output structure for :class:`WordSpace`."""

    q_base: torch.Tensor
    q_context: Optional[torch.Tensor]
    q_total: torch.Tensor
    dim_total: int


TelemetryPayload = Dict[str, float]
