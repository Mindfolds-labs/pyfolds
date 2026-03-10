"""Geometric resonance reasoning core for LEIBREG.

This module keeps distance-based attention for conceptual routing, but replaces
plain MLP feed-forward blocks with neuromorphic soma/dendrite dynamics inspired
by MPJRD principles used in the rest of PyFolds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from pyfolds.telemetry.types import TelemetryEvent


@dataclass
class REGState:
    """Optional recurrent state carried between calls.

    Attributes:
        membrane: membrane potential ``[depth, batch, seq, hidden_dim]``.
    """

    membrane: torch.Tensor


class ResonanceAttention(nn.Module):
    """Distance-kernel self-attention on tensors of shape ``[batch, seq, dim]``."""

    def __init__(
        self,
        dim: int,
        init_temperature: float = 0.1,
        eps: float = 1e-8,
        telemetry_collector: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.value = nn.Linear(dim, dim)
        self.log_temp = nn.Parameter(torch.tensor(float(init_temperature)).log())
        self.eps = eps
        self.telemetry_collector = telemetry_collector

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp_min(1e-5)

    def _emit(self, payload: dict[str, float]) -> None:
        if self.telemetry_collector is None:
            return
        try:
            self.telemetry_collector.emit(
                TelemetryEvent(0.0, "leibreg_resonance", "leibreg.reg_core", 0, payload)
            )
        except Exception:
            return

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must be [batch, seq, dim]")
        dist = torch.cdist(x, x, p=2)
        temp = self.temperature.to(dtype=x.dtype, device=x.device)
        kernel = 1.0 / (1.0 + (dist / temp) ** 2)
        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError("mask must be [batch, seq]")
            valid = (mask.unsqueeze(1) & mask.unsqueeze(2)).to(dtype=kernel.dtype)
            kernel = kernel * valid
        attn = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        out = torch.matmul(attn, self.value(x))

        entropy = -(attn * (attn.clamp_min(self.eps)).log()).sum(dim=-1).mean()
        self._emit(
            {
                "resonance_temperature": float(temp.mean().item()),
                "distance_mean": float(dist.mean().item()),
                "attention_entropy": float(entropy.item()),
            }
        )
        return out


class DendriticREGUnit(nn.Module):
    """Neuromorphic dendrite/soma unit used inside REG blocks.

    Steps:
    1. Route token embedding through multiple dendritic pathways.
    2. Compute local dendrite gates (optionally phase-modulated).
    3. Integrate at soma with homeostatic normalization.
    4. Update membrane potential with leaky integration.
    5. Apply saturating activation for spike-like bounded dynamics.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_dendrites: int = 4,
        membrane_decay: float = 0.9,
        homeostatic_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or n_dendrites <= 0:
            raise ValueError("input_dim, hidden_dim and n_dendrites must be > 0")
        if not (0.0 <= membrane_decay <= 1.0):
            raise ValueError("membrane_decay must be within [0, 1]")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_dendrites = n_dendrites
        self.membrane_decay = membrane_decay
        self.homeostatic_eps = homeostatic_eps

        self.dendrites = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Tanh(),
                )
                for _ in range(n_dendrites)
            ]
        )
        self.local_gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid()) for _ in range(n_dendrites)]
        )
        self.pathway_logits = nn.Parameter(torch.zeros(n_dendrites))
        self.soma_scale = nn.Parameter(torch.ones(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def _masked_softmax(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return torch.softmax(logits, dim=-1)
        mask = mask.to(dtype=torch.bool)
        very_neg = torch.finfo(logits.dtype).min
        masked_logits = torch.where(mask.unsqueeze(-1), logits, torch.full_like(logits, very_neg))
        weights = torch.softmax(masked_logits, dim=-1)
        return torch.where(mask.unsqueeze(-1), weights, torch.zeros_like(weights))

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        wave_phase: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("x must be [batch, seq, dim]")

        dendritic_outputs = []
        for dendrite, gate in zip(self.dendrites, self.local_gates):
            d_out = dendrite(x)
            d_gate = gate(x)
            if wave_phase is not None:
                d_gate = d_gate * (0.5 + 0.5 * torch.cos(wave_phase).unsqueeze(-1))
            dendritic_outputs.append(d_out * d_gate)

        stacked = torch.stack(dendritic_outputs, dim=2)

        pathway_logits = self.pathway_logits.view(1, 1, self.n_dendrites).expand(
            x.shape[0], x.shape[1], self.n_dendrites
        )
        pathway_weights = self._masked_softmax(pathway_logits, mask)

        soma_current = torch.sum(stacked * pathway_weights.unsqueeze(-1), dim=2)
        soma_current = soma_current * self.soma_scale

        centered = soma_current - soma_current.mean(dim=-1, keepdim=True)
        variance = centered.pow(2).mean(dim=-1, keepdim=True)
        normalized_current = centered / torch.sqrt(variance + self.homeostatic_eps)

        if state is None:
            state = torch.zeros_like(normalized_current)
        membrane = (self.membrane_decay * state) + normalized_current
        activation = torch.tanh(membrane)
        out = self.out_proj(activation)

        if mask is not None:
            mask_e = mask.unsqueeze(-1).to(dtype=out.dtype)
            out = out * mask_e
            membrane = membrane * mask_e

        return out, membrane


class REGBlock(nn.Module):
    """Pre-norm residual resonance block with neuromorphic integration."""

    def __init__(
        self,
        dim: int,
        ff_mult: int = 2,
        n_dendrites: int = 4,
        membrane_decay: float = 0.9,
        telemetry_collector: Optional[Any] = None,
    ) -> None:
        super().__init__()
        hidden = dim * ff_mult
        self.attn = ResonanceAttention(dim=dim, telemetry_collector=telemetry_collector)
        self.norm1 = nn.LayerNorm(dim)
        self.reg_unit = DendriticREGUnit(
            input_dim=dim,
            hidden_dim=hidden,
            n_dendrites=n_dendrites,
            membrane_decay=membrane_decay,
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        wave_phase: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.norm1(x + self.attn(x, mask=mask))
        reg_out, next_state = self.reg_unit(x, state=state, mask=mask, wave_phase=wave_phase)
        x = self.norm2(x + reg_out)
        return x, next_state


class REGCore(nn.Module):
    """Stack of :class:`REGBlock` layers with optional membrane recurrence.

    This is a hybrid conceptual reasoning core:
    - geometric resonance attention keeps LEIBREG relational dynamics,
    - dendritic pathways support modality routing,
    - membrane update adds temporal continuity across calls.
    """

    def __init__(
        self,
        dim: int = 4,
        depth: int = 2,
        n_dendrites: int = 4,
        ff_mult: int = 2,
        membrane_decay: float = 0.9,
        telemetry_collector: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be > 0")
        self.blocks = nn.ModuleList(
            [
                REGBlock(
                    dim=dim,
                    ff_mult=ff_mult,
                    n_dendrites=n_dendrites,
                    membrane_decay=membrane_decay,
                    telemetry_collector=telemetry_collector,
                )
                for _ in range(depth)
            ]
        )

    def _init_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [
            torch.zeros(x.shape[0], x.shape[1], block.reg_unit.hidden_dim, dtype=x.dtype, device=x.device)
            for block in self.blocks
        ]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        state: Optional[REGState] = None,
        wave_phase: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, REGState]:
        layer_states = self._init_state(x) if state is None else [state.membrane[i] for i in range(len(self.blocks))]

        next_layer_states: list[torch.Tensor] = []
        for block, layer_state in zip(self.blocks, layer_states):
            x, next_state = block(x, mask=mask, state=layer_state, wave_phase=wave_phase)
            next_layer_states.append(next_state)

        final_state = REGState(membrane=torch.stack(next_layer_states, dim=0))
        if return_state:
            return x, final_state
        return x


# Backward compatibility alias.
ProximityAttention = ResonanceAttention
