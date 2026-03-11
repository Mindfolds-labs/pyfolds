"""Integração dendrítica biologicamente motivada.

Substitui WTA hard por mecanismos biologicamente inspirados:

1. NMDA-like: sigmoid local por dendrito (threshold suave)
2. Shunting: normalização divisiva (competição suave)
3. bAP proporcional: contribuição por dendrito para reforço seletivo
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from .config import MPJRDConfig


class DendriticOutput(NamedTuple):
    """Output completo da integração dendrítica.

    Attributes:
        u: Potencial somático [B]
        v_nmda: Potenciais após gate NMDA [B, D]
        contribution: Contribuição proporcional por dendrito [B, D]
    """

    u: torch.Tensor
    v_nmda: torch.Tensor
    contribution: torch.Tensor
    gate_logit: torch.Tensor


class DendriticIntegration(nn.Module):
    """Integração dendrítica com NMDA local + shunting divisivo."""

    def __init__(self, cfg: MPJRDConfig):
        super().__init__()
        self.cfg = cfg
        self.n_dendrites = cfg.n_dendrites
        # Bias local do gate: inicializado em 0 para manter sigmoid(~0)=0.5 no início.
        self.gate_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def _theta_dend(self, theta: torch.Tensor) -> torch.Tensor:
        """Threshold local por dendrito = θ_soma × ratio."""
        if theta.dim() == 0:
            return theta * self.cfg.theta_dend_ratio
        return theta.mean() * self.cfg.theta_dend_ratio

    def forward(self, v_dend: torch.Tensor, theta: torch.Tensor) -> DendriticOutput:
        """Integra potenciais dendríticos com NMDA + shunting.

        Args:
            v_dend: Potenciais lineares [B, D]
            theta: Threshold somático atual
        """
        cfg = self.cfg
        D = self.n_dendrites

        theta_dend = self._theta_dend(theta)
        raw_gate_logit = v_dend - theta_dend

        # Centralização/escala estritamente espacial por amostra (dim=1).
        local_mu = raw_gate_logit.mean(dim=1, keepdim=True)
        local_sigma = raw_gate_logit.std(dim=1, keepdim=True, unbiased=False).clamp_min(cfg.eps)
        gate_logit = ((raw_gate_logit - local_mu) / local_sigma) + self.gate_bias.to(raw_gate_logit.dtype)

        v_nmda = torch.sigmoid(cfg.dendrite_gain * gate_logit)

        sum_dend = v_nmda.sum(dim=1, keepdim=True)
        v_norm = v_nmda / (cfg.shunting_eps + cfg.shunting_strength * sum_dend)

        total = v_norm.sum(dim=1, keepdim=True).clamp_min(cfg.eps)
        contribution = v_norm / total

        u = D * v_norm.sum(dim=1)
        return DendriticOutput(
            u=u,
            v_nmda=v_nmda,
            contribution=contribution,
            gate_logit=gate_logit,
        )

    def extra_repr(self) -> str:
        cfg = self.cfg
        return (
            f"D={self.n_dendrites}, "
            f"gain={cfg.dendrite_gain}, "
            f"theta_dend_ratio={cfg.theta_dend_ratio}, "
            f"shunting_eps={cfg.shunting_eps}, "
            f"shunting_strength={cfg.shunting_strength}"
        )
