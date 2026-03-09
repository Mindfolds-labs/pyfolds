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

        init_l = self._n_to_l(init_n)
        self.register_buffer("L", torch.tensor([init_l], dtype=torch.int32))
        self._sync_quantized_state_from_initialization()

        # Potencial interno
        self.register_buffer("I", torch.zeros(1, dtype=torch.float32))

        # Proteção contra saturação
        self.register_buffer("protection", torch.tensor([False], dtype=torch.bool))
        self.register_buffer("sat_time", torch.zeros(1, dtype=torch.float32))

        # Traços para consolidação (Hebb + STDP separados)
        self.register_buffer("eligibility", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("stdp_eligibility", torch.zeros(1, dtype=torch.float32))

        # Estado de curto prazo (u, R) para compatibilidade
        self.register_buffer("u", torch.tensor([cfg.u0], dtype=torch.float32))
        self.register_buffer("R", torch.tensor([cfg.R0], dtype=torch.float32))

    def _is_uniform_mode(self) -> bool:
        """Retorna True quando a quantização ativa usa níveis lineares em W."""
        return self.cfg.weight_quantization == "uniformW"

    @property
    def _l_min(self) -> int:
        return 0

    @property
    def _l_max(self) -> int:
        return self.cfg.n_levels - 1

    def _n_to_l(self, n_value: int) -> int:
        """Mapeia N para o nível discreto L de forma proporcional."""
        if self.cfg.n_max <= 0:
            return self._l_min
        scaled = (float(n_value) / float(self.cfg.n_max)) * float(self._l_max)
        return int(round(min(max(scaled, self._l_min), self._l_max)))

    def _l_to_n(self, l_value: int) -> int:
        """Mapeia L para N para manter telemetria compatível."""
        scaled = (float(l_value) / float(max(1, self._l_max))) * float(self.cfg.n_max)
        return int(round(min(max(scaled, self.cfg.n_min), self.cfg.n_max)))

    def _sync_quantized_state_from_initialization(self) -> None:
        """Sincroniza buffers N/L após inicialização."""
        if self._is_uniform_mode():
            self._sync_n_from_l()
            return

        self.L.fill_(self._n_to_l(int(self.N.item())))

    def _sync_n_from_l(self) -> None:
        """Mantém ``N`` coerente com ``L`` para telemetria/compatibilidade."""
        self.N.fill_(self._l_to_n(int(self.L.item())))

    def _current_level(self) -> int:
        """Retorna o estado discreto ativo da sinapse."""
        if self._is_uniform_mode():
            return int(self.L.item())
        return int(self.N.item())

    def _set_current_level(self, level: int) -> None:
        """Atualiza o estado discreto ativo com clamp por modo."""
        if self._is_uniform_mode():
            self.L.fill_(level)
            self.L.clamp_(self._l_min, self._l_max)
            self._sync_n_from_l()
            return

        self.N.fill_(level)
        self.N.clamp_(self.cfg.n_min, self.cfg.n_max)
        self.L.fill_(self._n_to_l(int(self.N.item())))

    def _weight_from_current_state(self) -> torch.Tensor:
        """Computa o peso efetivo a partir do estado discreto ativo."""
        if self._is_uniform_mode():
            levels_minus_one = max(1, self.cfg.n_levels - 1)
            return (self.L.float() / float(levels_minus_one)) * self.cfg.w_max

        return safe_weight_law(
            self.N,
            w_scale=self.cfg.w_scale,
            max_log_val=self.cfg.max_log_weight,
            enforce_checks=self.cfg.numerical_stability_checks,
        )

    @torch.no_grad()
    def _apply_weight_step(self, delta: int) -> None:
        """Aplica passo discreto no estado de peso conforme o modo ativo."""
        self._set_current_level(self._current_level() + int(delta))

    def _at_upper_bound(self) -> bool:
        """Indica se o estado discreto já está saturado no máximo."""
        upper = self._l_max if self._is_uniform_mode() else self.cfg.n_max
        return self._current_level() >= upper

    def _at_lower_bound(self) -> bool:
        """Indica se o estado discreto já está saturado no mínimo."""
        lower = self._l_min if self._is_uniform_mode() else self.cfg.n_min
        return self._current_level() <= lower

    @property
    def W(self) -> torch.Tensor:
        """
        Peso sináptico derivado do número de filamentos (Bartol Log Law).

        Implementa a relação:
            W = log2(1 + N) / w_scale

        Returns:
            Tensor escalar com o peso derivado em ponto flutuante.
        """
        return self._weight_from_current_state()

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
    def _sanitize_state_buffers(self) -> None:
        """Recupera buffers da sinapse após corrupção transitória (ex.: bitflip ECC)."""
        finite_i = torch.nan_to_num(
            self.I, nan=0.0, posinf=self.cfg.i_max, neginf=self.cfg.i_min
        )
        self.I.copy_(finite_i.clamp(self.cfg.i_min, self.cfg.i_max))

        finite_elig = torch.nan_to_num(
            self.eligibility,
            nan=0.0,
            posinf=self.cfg.max_eligibility,
            neginf=-self.cfg.max_eligibility,
        )
        self.eligibility.copy_(finite_elig.clamp(-self.cfg.max_eligibility, self.cfg.max_eligibility))

        finite_stdp_elig = torch.nan_to_num(
            self.stdp_eligibility,
            nan=0.0,
            posinf=self.cfg.max_eligibility,
            neginf=-self.cfg.max_eligibility,
        )
        self.stdp_eligibility.copy_(
            finite_stdp_elig.clamp(-self.cfg.max_eligibility, self.cfg.max_eligibility)
        )

        finite_sat = torch.nan_to_num(
            self.sat_time, nan=0.0, posinf=self.cfg.saturation_recovery_time, neginf=0.0
        )
        self.sat_time.copy_(finite_sat.clamp_min(0.0))

        clamped_n = self.N.to(dtype=torch.int32).clamp(self.cfg.n_min, self.cfg.n_max)
        self.N.copy_(clamped_n)
        clamped_l = self.L.to(dtype=torch.int32).clamp(self._l_min, self._l_max)
        self.L.copy_(clamped_l)

        if self._is_uniform_mode():
            self._sync_n_from_l()
        else:
            self.L.fill_(self._n_to_l(int(self.N.item())))

    @torch.no_grad()
    def update(
        self,
        pre_rate: torch.Tensor,
        post_rate: torch.Tensor,
        R: torch.Tensor,
        dt: float = 1.0,
        mode: Optional[LearningMode] = None,
    ) -> None:
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

        self._sanitize_state_buffers()

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
            if not self._at_upper_bound():
                self._apply_weight_step(+1)
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
            if not self._at_lower_bound():
                self._apply_weight_step(-1)
                self.I.zero_()

                # Se estava protegido e saiu da saturação
                if (not self._at_upper_bound()) and self.protection.item():
                    self.protection.fill_(False)
                    self.sat_time.zero_()

        # ===== Recuperação da saturação =====
        # Atualização vetorial/máscara para evitar sync D2H via `.item()`.
        self.sat_time.add_(self.protection.to(dtype=self.sat_time.dtype) * dt)
        if cfg.saturation_recovery_time > 0:
            recovery_mask = self.protection & (
                self.sat_time >= cfg.saturation_recovery_time
            )
            self.protection.masked_fill_(recovery_mask, False)
            self.sat_time.masked_fill_(recovery_mask, 0.0)

    @torch.no_grad()
    def _sync_distributed_plasticity_state(self) -> None:
        """Sincroniza buffers plásticos após consolidação em execução distribuída."""
        if not self.cfg.distributed_sync_on_consolidate:
            return

        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return

        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return

        if self._is_uniform_mode():
            l_float = self.L.to(dtype=torch.float32)
            torch.distributed.all_reduce(l_float, op=torch.distributed.ReduceOp.SUM)
            l_float.div_(float(world_size))
            self.L.copy_(torch.round(l_float).to(dtype=self.L.dtype))
            self.L.clamp_(self._l_min, self._l_max)
            self.N.fill_(self._l_to_n(int(self.L.item())))
        else:
            n_float = self.N.to(dtype=torch.float32)
            torch.distributed.all_reduce(n_float, op=torch.distributed.ReduceOp.SUM)
            n_float.div_(float(world_size))
            self.N.copy_(torch.round(n_float).to(dtype=self.N.dtype))
            self.N.clamp_(self.cfg.n_min, self.cfg.n_max)
            self.L.fill_(self._n_to_l(int(self.N.item())))

        torch.distributed.all_reduce(self.I, op=torch.distributed.ReduceOp.SUM)
        self.I.div_(float(world_size))

        torch.distributed.all_reduce(
            self.eligibility, op=torch.distributed.ReduceOp.SUM
        )
        self.eligibility.div_(float(world_size))

        torch.distributed.all_reduce(
            self.stdp_eligibility, op=torch.distributed.ReduceOp.SUM
        )
        self.stdp_eligibility.div_(float(world_size))

        protection_float = self.protection.to(dtype=torch.float32)
        torch.distributed.all_reduce(
            protection_float, op=torch.distributed.ReduceOp.SUM
        )
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
        dt_abs = abs(dt)
        consolidation_scale = self.cfg.consolidation_rate * (dt_abs / self.cfg.tau_consolidation)
        combined_eligibility = self.eligibility + (
            self.stdp_eligibility * self.cfg.stdp_consolidation_scale
        )
        transfer = combined_eligibility.to(dtype=torch.float32) * consolidation_scale
        delta_n = torch.round(transfer).to(dtype=self.N.dtype)

        # 2. Adição Maciça In-Place
        # Adicionar +0 na GPU é exponencialmente mais barato que sincronizar um torch.any()
        if self._is_uniform_mode():
            delta_l = torch.round(transfer).to(dtype=self.L.dtype)
            self.L.add_(delta_l)
            self.L.clamp_(self._l_min, self._l_max)
            self.N.fill_(self._l_to_n(int(self.L.item())))
        else:
            self.N.add_(delta_n)
            self.N.clamp_(self.cfg.n_min, self.cfg.n_max)
            self.L.fill_(self._n_to_l(int(self.N.item())))

        # 3. Reseta elegibilidade massivamente
        self.eligibility.zero_()
        self.stdp_eligibility.zero_()

        # 4. Decaimento natural de I
        if self.I.numel() > 0:
            self.I.mul_(self.cfg.i_decay_sleep)

        self._sync_distributed_plasticity_state()

    def get_state(self) -> dict:
        """Retorna estado completo da sinapse (sem .item() para tensores)."""
        return {
            "N": self.N.clone(),
            "L": self.L.clone(),
            "I": self.I.clone(),
            "W": self.W.clone(),
            "protection": self.protection.clone(),
            "sat_time": self.sat_time.clone(),
            "eligibility": self.eligibility.clone(),
            "stdp_eligibility": self.stdp_eligibility.clone(),
        }

    def load_state(self, state: dict) -> None:
        """Carrega estado na sinapse."""
        if "N" in state:
            self.N.data = state["N"].to(self.N.device)
        if "L" in state:
            self.L.data = state["L"].to(self.L.device)
        if "I" in state:
            self.I.data = state["I"].to(self.I.device)
        if "protection" in state:
            self.protection.data = state["protection"].to(self.protection.device)
        if "sat_time" in state:
            self.sat_time.data = state["sat_time"].to(self.sat_time.device)
        if "eligibility" in state:
            self.eligibility.data = state["eligibility"].to(self.eligibility.device)
        if "stdp_eligibility" in state:
            self.stdp_eligibility.data = state["stdp_eligibility"].to(self.stdp_eligibility.device)

        self._sanitize_state_buffers()

    def extra_repr(self) -> str:
        """Representação string da sinapse."""
        return (
            f"N={self.N.item()}, L={self.L.item()}, I={self.I.item():.2f}, "
            f"W={self.W.item():.3f}, prot={self.protection.item()}"
        )
