# pyfolds/core/synapse.py
"""
Sinapse MPJRD - unidade fundamental de plasticidade - VERSÃO OTIMIZADA

Características:
- N: filamentos (0-31) - MEMÓRIA DE LONGO PRAZO
- I: potencial interno - MEMÓRIA DE CURTO PRAZO
- W: peso derivado = log2(1+N)/w_scale
- protection: modo de proteção contra saturação
- sat_time: tempo em saturação
- eligibility: traço para consolidação two-factor

✅ OTIMIZAÇÕES:
    - Trabalha com tensores (evita .item() em loops críticos)
    - Operações in-place para eficiência
    - Suporte a learning_rate_multiplier por modo
    - Usa constantes da config
    - ✅ Filtro activity_threshold na plasticidade
    - ✅ Uso minimizado de .item()
"""

import torch
import torch.nn as nn
from typing import Optional
from .config import MPJRDConfig
from ..utils.types import LearningMode
from ..utils.math import clamp_rate, clamp_R


class MPJRDSynapse(nn.Module):
    """
    Sinapse MPJRD com plasticidade three-factor e two-factor consolidation.
    
    ✅ OTIMIZADO: trabalha com tensores diretamente
    ✅ CONFIGURÁVEL: usa constantes da config
    ✅ MODOS: suporta multiplicador de learning rate por modo
    """

    def __init__(self, cfg: MPJRDConfig, init_n: Optional[int] = None):
        super().__init__()
        self.cfg = cfg

        # Inicialização do número de filamentos
        if init_n is None:
            init_n = int(torch.randint(cfg.n_min, min(5, cfg.n_max + 1), (1,)).item())
        self.register_buffer("N", torch.tensor([init_n], dtype=torch.int32))

        # Potencial interno
        self.register_buffer("I", torch.zeros(1, dtype=torch.float32))

        # Proteção contra saturação
        self.register_buffer("protection", torch.tensor([False], dtype=torch.bool))
        self.register_buffer("sat_time", torch.zeros(1, dtype=torch.float32))

        # Traço para consolidação two-factor
        self.register_buffer("eligibility", torch.zeros(1, dtype=torch.float32))

        # Estado de curto prazo (u, R) para compatibilidade
        self.register_buffer("u", torch.tensor([cfg.u0], dtype=torch.float32))
        self.register_buffer("R", torch.tensor([cfg.R0], dtype=torch.float32))

    @property
    def W(self) -> torch.Tensor:
        """Peso sináptico derivado do número de filamentos."""
        return torch.log2(1.0 + self.N.float()) / self.cfg.w_scale

    @torch.no_grad()
    def update(self, 
               pre_rate: torch.Tensor, 
               post_rate: torch.Tensor,
               R: torch.Tensor, 
               dt: float = 1.0,
               mode: Optional[LearningMode] = None) -> None:
        """
        Atualização baseada em taxas (Three-factor learning rule).
        
        ✅ OTIMIZADO: trabalha com tensores, evita .item() no caminho crítico
        ✅ MODOS: aplica multiplicador de learning rate baseado no modo
        ✅ FILTRO: aplica activity_threshold para sinapses inativas
        
        Args:
            pre_rate: Taxa pré-sináptica [B] ou [1]
            post_rate: Taxa pós-sináptica [1]
            R: Sinal neuromodulador [1]
            dt: Passo de tempo (ms)
            mode: Modo de aprendizado (para multiplicador de LR)
        """
        if not self.cfg.plastic:
            return

        cfg = self.cfg
        dt = abs(dt)

        # ===== APLICA MULTIPLICADOR DE LEARNING RATE =====
        if mode is not None:
            lr_mult = mode.learning_rate_multiplier
            effective_eta = cfg.i_eta * lr_mult
        else:
            effective_eta = cfg.i_eta

        # Normaliza entradas
        pre_rate = clamp_rate(pre_rate)  # [B] ou [1]
        post_rate = clamp_rate(post_rate)  # [1]
        R = clamp_R(R)  # [1]

        # ✅ FILTRO: ignora sinapses inativas
        # Se pre_rate < activity_threshold, considera como 0 para plasticidade
        active_mask = (pre_rate > cfg.activity_threshold).float()  # [B]
        pre_rate_filtered = pre_rate * active_mask  # [B]

        # Hebb = pre * post (broadcast)
        # post_rate é [1], pre_rate_filtered é [B] → resultado [B]
        hebb = (pre_rate_filtered * post_rate).clamp(0.0, 1.0)  # [B]

        # Ganho dependente do peso (escalar)
        W_val = self.W.item()  # Único .item() necessário
        gain = 1.0 + cfg.beta_w * W_val

        # Delta de I (versão normalizada)
        # R é [1], neuromod_scale é escalar
        delta = effective_eta * (R * cfg.neuromod_scale) * hebb * gain * dt  # [B]

        # ✅ Atualização in-place (tensores)
        # Usa média do batch para atualizar I (sinapse única)
        delta_mean = delta.mean()  # []
        self.I.mul_(cfg.i_gamma).add_(delta_mean)
        self.I.clamp_(cfg.i_min, cfg.i_max)

        # ✅ Eligibility in-place
        self.eligibility.add_(delta_mean)

        # ===== LTP (Promoção) =====
        # Usa comparação tensorial, evita .item() no caminho crítico
        if self.I >= cfg.i_ltp_th:
            if self.N < cfg.n_max:
                self.N.add_(1)
                self.I.zero_()
                self.protection.fill_(False)
                self.sat_time.zero_()
            else:
                # Saturado: entra em modo de proteção
                self.protection.fill_(True)
                self.I.zero_()
                self.sat_time.zero_()

        # ===== LTD (Demoção) =====
        # Determina threshold baseado em proteção (precisa de .item() para condicional)
        ltd_th = cfg.ltd_threshold_saturated if self.protection.item() else cfg.i_ltd_th
        
        if self.I <= ltd_th:
            if self.N > cfg.n_min:
                self.N.add_(-1)
                self.I.zero_()
                
                # Se estava protegido e saiu da saturação
                if self.N.item() == cfg.n_max - 1 and self.protection.item():
                    self.protection.fill_(False)
                    self.sat_time.zero_()

        # ===== Recuperação da saturação =====
        if self.protection.item():
            self.sat_time.add_(dt)
            if self.sat_time.item() >= cfg.saturation_recovery_time:
                self.protection.fill_(False)
                self.sat_time.zero_()

    @torch.no_grad()
    def consolidate(self, dt: float = 1.0) -> None:
        """
        Consolida o fator volátil em estável (two-factor consolidation).
        
        ✅ OTIMIZADO: usa i_decay_sleep da config
        """
        # Verifica se há elegibilidade
        if self.eligibility.numel() == 0 or self.eligibility.item() == 0:
            return
        
        # Transfere elegibilidade para N
        transfer = self.eligibility.item() * self.cfg.consolidation_rate * abs(dt)
        delta_n = int(round(transfer))
        
        if delta_n != 0:
            self.N.add_(delta_n)
            self.N.clamp_(self.cfg.n_min, self.cfg.n_max)

        # Reseta elegibilidade
        self.eligibility.zero_()

        # ✅ Decaimento natural de I
        if self.I.numel() > 0:
            self.I.mul_(self.cfg.i_decay_sleep)

    def get_state(self) -> dict:
        """Retorna estado completo da sinapse (sem .item() para tensores)."""
        return {
            'N': self.N.clone(),
            'I': self.I.clone(),
            'W': self.W.clone(),
            'protection': self.protection.clone(),
            'sat_time': self.sat_time.clone(),
            'eligibility': self.eligibility.clone(),
        }

    def load_state(self, state: dict) -> None:
        """Carrega estado na sinapse."""
        if 'N' in state:
            self.N.data = state['N'].to(self.N.device)
        if 'I' in state:
            self.I.data = state['I'].to(self.I.device)
        if 'protection' in state:
            self.protection.data = state['protection'].to(self.protection.device)
        if 'sat_time' in state:
            self.sat_time.data = state['sat_time'].to(self.sat_time.device)
        if 'eligibility' in state:
            self.eligibility.data = state['eligibility'].to(self.eligibility.device)

    def extra_repr(self) -> str:
        """Representação string da sinapse."""
        return (f"N={self.N.item()}, I={self.I.item():.2f}, "
                f"W={self.W.item():.3f}, prot={self.protection.item()}")