"""Backend vetorizado de sinapses para dendritos MPJRD."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import MPJRDConfig
from ..utils.types import LearningMode


class VectorizedSynapseBatch(nn.Module):
    """Representa todas as sinapses de um dendrito como tensores ``[S]``.

    Parameters
    ----------
    cfg : MPJRDConfig
        Configuração global do neurônio.
    n_synapses : int
        Quantidade de sinapses no dendrito.
    """

    def __init__(self, cfg: MPJRDConfig, n_synapses: int):
        super().__init__()
        self.cfg = cfg
        self.S = int(n_synapses)

        low = max(cfg.n_min, cfg.n_max // 4)
        high = max(low + 1, min(cfg.n_max + 1, (3 * cfg.n_max) // 4 + 1))
        n_init = torch.randint(low, high, (self.S,), dtype=torch.int32)

        self.register_buffer("N", n_init)
        self.register_buffer("I", torch.zeros(self.S, dtype=torch.float32))
        self.register_buffer("L", torch.round((n_init.float() / max(1, cfg.n_max)) * (cfg.n_levels - 1)).to(torch.int32))
        self.register_buffer("protection", torch.zeros(self.S, dtype=torch.bool))
        self.register_buffer("sat_time", torch.zeros(self.S, dtype=torch.float32))
        self.register_buffer("eligibility", torch.zeros(self.S, dtype=torch.float32))
        self.register_buffer("stdp_eligibility", torch.zeros(self.S, dtype=torch.float32))
        self.register_buffer("u", torch.full((self.S,), float(cfg.u0), dtype=torch.float32))
        self.register_buffer("R", torch.full((self.S,), float(cfg.R0), dtype=torch.float32))

    @property
    def W(self) -> torch.Tensor:
        """Pesos sinápticos vetorizados ``[S]``."""
        return torch.log2(1.0 + self.N.float()) / self.cfg.w_scale

    @torch.no_grad()
    def update_batch(
        self,
        pre_rate: torch.Tensor,
        post_rate: torch.Tensor,
        R: torch.Tensor,
        dt: float = 1.0,
        mode: Optional[LearningMode] = None,
    ) -> None:
        """Atualiza plasticidade three-factor para todas as sinapses em lote.

        Parameters
        ----------
        pre_rate : torch.Tensor
            Atividade pré-sináptica com shape ``[B, S]``.
        post_rate : torch.Tensor
            Atividade pós-sináptica com shape ``[B]`` ou escalar.
        R : torch.Tensor
            Reforço neuromodulador escalar.
        dt : float, default=1.0
            Passo temporal.
        mode : Optional[LearningMode], default=None
            Modo de aprendizado para escalonar learning rate.
        """
        cfg = self.cfg
        lr_mult = mode.learning_rate_multiplier if mode is not None else 1.0
        effective_eta = cfg.i_eta * lr_mult

        active = (pre_rate > cfg.activity_threshold).float()
        pre_f = (pre_rate * active).clamp(0.0, 1.0)

        post_b = post_rate.reshape(-1, 1).expand_as(pre_f).clamp(0.0, 1.0)
        r_scaled = torch.tanh((R * cfg.neuromod_scale).clamp(-1.0, 1.0))
        r_pos = r_scaled.clamp(min=0.0)
        r_neg = (-r_scaled).clamp(min=0.0)

        ltp = cfg.A_plus * (pre_f * post_b) * r_pos
        ltd = cfg.A_minus * cfg.hebbian_ltd_ratio * (pre_f * (1.0 - post_b)) * (r_pos + r_neg)

        gain = 1.0 + cfg.beta_w * self.W.unsqueeze(0)
        delta = effective_eta * (ltp - ltd) * gain * abs(float(dt))
        delta_mean = delta.mean(dim=0)

        self.I.mul_(cfg.i_gamma).add_(delta_mean).clamp_(cfg.i_min, cfg.i_max)
        self.eligibility.add_(delta_mean)

        ltp_mask = (self.I >= cfg.i_ltp_th) & ~self.protection
        at_max = self.N >= cfg.n_max
        self.N[ltp_mask & ~at_max] += 1
        self.protection[ltp_mask & at_max] = True
        self.I[ltp_mask] = 0.0
        self.sat_time[ltp_mask & at_max] = 0.0

        ltd_th = torch.where(
            self.protection,
            torch.full_like(self.I, cfg.ltd_threshold_saturated),
            torch.full_like(self.I, cfg.i_ltd_th),
        )
        ltd_mask = self.I <= ltd_th
        at_min = self.N <= cfg.n_min
        self.N[ltd_mask & ~at_min] -= 1
        self.I[ltd_mask] = 0.0

        exited = ltd_mask & self.protection & ~at_max
        self.protection[exited] = False
        self.sat_time[exited] = 0.0

        self.sat_time.add_(self.protection.float() * abs(float(dt)))
        if cfg.saturation_recovery_time > 0:
            recovered = self.protection & (self.sat_time >= cfg.saturation_recovery_time)
            self.protection[recovered] = False
            self.sat_time[recovered] = 0.0

        self.N.clamp_(cfg.n_min, cfg.n_max)
        self.L.copy_(
            torch.round(self.N.float() / float(max(1, cfg.n_max)) * float(cfg.n_levels - 1)).to(self.L.dtype)
        )

    @torch.no_grad()
    def consolidate(self, dt: float = 1.0) -> None:
        """Consolida elegibilidade em estado discreto vetorial."""
        if self.eligibility.numel() == 0:
            return
        scale = self.cfg.consolidation_rate * (abs(float(dt)) / self.cfg.tau_consolidation)
        combined = self.eligibility + self.stdp_eligibility * self.cfg.stdp_consolidation_scale
        delta = torch.round(combined.float() * scale).to(self.N.dtype)
        self.N.add_(delta).clamp_(self.cfg.n_min, self.cfg.n_max)
        self.L.copy_(
            torch.round(self.N.float() / float(max(1, self.cfg.n_max)) * float(self.cfg.n_levels - 1)).to(self.L.dtype)
        )
        self.eligibility.zero_()
        self.stdp_eligibility.zero_()
        self.I.mul_(self.cfg.i_decay_sleep)
