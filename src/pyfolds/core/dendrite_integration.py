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
        # Threshold local por dendrito para manter logit próximo da região ativa.
        self.theta = nn.Parameter(0.01 * torch.randn(self.n_dendrites, dtype=torch.float32))

    def _theta_dend(self, theta: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Threshold local por dendrito = θ_soma × ratio."""
        if theta.dim() == 0:
            theta_eff = theta * self.cfg.theta_dend_ratio
        else:
            theta_eff = theta.mean() * self.cfg.theta_dend_ratio
        theta_eff = theta_eff + self.theta.to(dtype=dtype, device=device)
        return theta_eff.unsqueeze(0)

    def forward(self, v_dend: torch.Tensor, theta: torch.Tensor) -> DendriticOutput:
        """Integra potenciais dendríticos com NMDA + shunting.

        Args:
            v_dend: Potenciais lineares [B, D]
            theta: Threshold somático atual
        """
        cfg = self.cfg
        D = self.n_dendrites

        theta_eff = self._theta_dend(theta, dtype=v_dend.dtype, device=v_dend.device)
        raw_logit = v_dend - theta_eff

        local_mu = raw_logit.mean(dim=-1, keepdim=True)
        local_sigma = raw_logit.std(dim=-1, keepdim=True, unbiased=False).clamp_min(cfg.gate_local_norm_eps)
        norm_logit = (raw_logit - local_mu) / local_sigma
        gate_logit = cfg.gate_logit_scale * norm_logit

        v_nmda = torch.sigmoid(gate_logit)

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
            f"gate_logit_scale={cfg.gate_logit_scale}, "
            f"gate_local_norm_eps={cfg.gate_local_norm_eps}, "
            f"theta_dend_ratio={cfg.theta_dend_ratio}, "
            f"shunting_eps={cfg.shunting_eps}, "
            f"shunting_strength={cfg.shunting_strength}"
        )
