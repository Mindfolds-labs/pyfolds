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
from ..utils.math import clamp_rate, clamp_R, safe_weight_law

__all__ = ["MPJRDSynapse"]


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
            low = max(cfg.n_min, cfg.n_max // 4)
            high = max(low + 1, min(cfg.n_max + 1, (3 * cfg.n_max) // 4 + 1))
            init_n = int(torch.randint(low, high, (1,)).item())
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
        """
        Peso sináptico derivado do número de filamentos (Bartol Log Law).

        Implementa a relação:
            W = log2(1 + N) / w_scale

        Returns:
            Tensor escalar com o peso derivado em ponto flutuante.
        """
        return safe_weight_law(
            self.N,
            w_scale=self.cfg.w_scale,
            max_log_val=self.cfg.max_log_weight,
            enforce_checks=self.cfg.numerical_stability_checks,
        )

    @torch.no_grad()
    def _update_with_soft_saturation(self, delta_mean: torch.Tensor) -> None:
        """Atualiza I com amortecimento suave perto dos limites."""
        cfg = self.cfg

        near_min = (self.I - cfg.i_min) < 1.0
        near_max = (cfg.i_max - self.I) < 1.0
        damped_delta = delta_mean

        if near_max.any() and damped_delta.item() > 0:
            excess = (self.I[near_max] - cfg.i_max) / 5.0
            damping = 1.0 / (1.0 + torch.exp(excess))
            damped_delta = damped_delta * damping.mean()

        if near_min.any() and damped_delta.item() < 0:
            deficit = (cfg.i_min - self.I[near_min]) / 5.0
            damping = 1.0 / (1.0 + torch.exp(deficit))
            damped_delta = damped_delta * damping.mean()

        self.I.mul_(cfg.i_gamma).add_(damped_delta)
        self.I.clamp_(cfg.i_min, cfg.i_max)

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
        pre_rate = clamp_rate(pre_rate).to(self.I.device)  # [B] ou [1]
        post_rate = clamp_rate(post_rate).to(self.I.device)  # [1]
        R = clamp_R(R).to(self.I.device)  # [1]

        # ✅ FILTRO: ignora sinapses inativas
        # Se pre_rate < activity_threshold, considera como 0 para plasticidade
        active_mask = (pre_rate > cfg.activity_threshold).float()  # [B]
        pre_rate_filtered = pre_rate * active_mask  # [B]

        # Hebb LTP = pre * post (broadcast)
        # post_rate é [1], pre_rate_filtered é [B] → resultado [B]
        hebb_ltp = (pre_rate_filtered * post_rate).clamp(0.0, 1.0)  # [B]

        # Hebb LTD explícito: pre * (1 - post)
        hebb_ltd = (pre_rate_filtered * (1.0 - post_rate)).clamp(0.0, 1.0)

        # Neuromodulação robusta com separação de vias LTP/LTD.
        # R>0 favorece LTP, R<0 favorece LTD.
        # O clamp/tanh evita explosões numéricas quando o sinal externo oscila.
        r_scaled = torch.tanh((R * cfg.neuromod_scale).clamp(-1.0, 1.0))
        r_pos = torch.clamp(r_scaled, min=0.0, max=1.0)
        r_neg = torch.clamp(-r_scaled, min=0.0, max=1.0)

        ltp_drive = cfg.A_plus * hebb_ltp * r_pos
        ltd_drive = cfg.A_minus * cfg.hebbian_ltd_ratio * hebb_ltd * (r_pos + r_neg)
        hebb_effective = ltp_drive - ltd_drive

        # Ganho dependente do peso (tensor escalar)
        gain = 1.0 + cfg.beta_w * self.W

        # Delta de I (versão normalizada)
        delta = effective_eta * hebb_effective * gain * dt  # [B]

        # ✅ Atualização in-place (tensores)
        # Usa média do batch para atualizar I (sinapse única)
        if delta.numel() > 0:
            delta_mean = delta.mean()  # []
        else:
            delta_mean = torch.tensor(0.0, device=self.I.device)
        self._update_with_soft_saturation(delta_mean=delta_mean)

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
        # Threshold tensorial para LTD (evita branching por .item())
        ltd_th = torch.where(
            self.protection,
            torch.full_like(self.I, cfg.ltd_threshold_saturated),
            torch.full_like(self.I, cfg.i_ltd_th),
        )
        
        if self.I <= ltd_th:
            if self.N > cfg.n_min:
                self.N.add_(-1)
                self.I.zero_()
                
                # Se estava protegido e saiu da saturação
                if self.N.item() == cfg.n_max - 1 and self.protection.item():
                    self.protection.fill_(False)
                    self.sat_time.zero_()

        # ===== Recuperação da saturação =====
        # Atualização vetorial/máscara para evitar sync D2H via `.item()`.
        self.sat_time.add_(self.protection.to(dtype=self.sat_time.dtype) * dt)
        if cfg.saturation_recovery_time > 0:
            recovery_mask = self.protection & (self.sat_time >= cfg.saturation_recovery_time)
            self.protection.masked_fill_(recovery_mask, False)
            self.sat_time.masked_fill_(recovery_mask, 0.0)

    @torch.no_grad()
    def _sync_distributed_plasticity_state(self) -> None:
        """Sincroniza buffers plásticos após consolidação em execução distribuída."""
        if not self.cfg.distributed_sync_on_consolidate:
            return

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return

        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return

        n_float = self.N.to(dtype=torch.float32)
        torch.distributed.all_reduce(n_float, op=torch.distributed.ReduceOp.SUM)
        n_float.div_(float(world_size))
        self.N.copy_(torch.round(n_float).to(dtype=self.N.dtype))

        torch.distributed.all_reduce(self.I, op=torch.distributed.ReduceOp.SUM)
        self.I.div_(float(world_size))

        torch.distributed.all_reduce(self.eligibility, op=torch.distributed.ReduceOp.SUM)
        self.eligibility.div_(float(world_size))

        protection_float = self.protection.to(dtype=torch.float32)
        torch.distributed.all_reduce(protection_float, op=torch.distributed.ReduceOp.SUM)
        self.protection.copy_(protection_float >= (world_size * 0.5))

        torch.distributed.all_reduce(self.sat_time, op=torch.distributed.ReduceOp.SUM)
        self.sat_time.div_(float(world_size))

    @torch.no_grad()
    def consolidate(self, dt: float = 1.0) -> None:
        """
        Consolida o fator volátil em estável (two-factor consolidation).

        ✅ OTIMIZADO V2: Remoção TOTAL de sincronização D2H.
        Matemática in-place pura, sem desvios baseados no conteúdo da VRAM.
        """
        # .numel() é seguro pois lê apenas o metadado (shape) alocado na CPU Host
        if self.eligibility.numel() == 0:
            return

        # 1. Matemática Vetorial sem Condicionais
        # Se a elegibilidade for 0, delta_n será calculado como +0 automaticamente.
        transfer = self.eligibility.to(dtype=torch.float32) * (self.cfg.consolidation_rate * abs(dt))
        delta_n = torch.round(transfer).to(dtype=self.N.dtype)

        # 2. Adição Maciça In-Place
        # Adicionar +0 na GPU é exponencialmente mais barato que sincronizar um torch.any()
        self.N.add_(delta_n)
        self.N.clamp_(self.cfg.n_min, self.cfg.n_max)

        # 3. Reseta elegibilidade massivamente
        self.eligibility.zero_()

        # 4. Decaimento natural de I
        if self.I.numel() > 0:
            self.I.mul_(self.cfg.i_decay_sleep)

        self._sync_distributed_plasticity_state()

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
