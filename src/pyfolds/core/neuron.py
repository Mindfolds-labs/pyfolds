# pyfolds/core/neuron.py
"""
Neurônio MPJRD completo - Versão Final Otimizada com LOGGING

Características:
- ✅ Buffers in-place para segurança
- ✅ Acumulação unificada (StatisticsAccumulator)
- ✅ Telemetria integrada nos boundaries
- ✅ Modos de aprendizado (ONLINE, BATCH, SLEEP, INFERENCE)
- ✅ Exportação de métricas
- ✅ Suporte a learning_rate_multiplier via mode
- ✅ Usa constantes da config (activity_threshold)
- ✅ LOGGING profissional (DEBUG, INFO, WARNING, TRACE)
- ✅ Validação de devices
- ✅ Callbacks de homeostase
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from .config import MPJRDConfig
from .base import BaseNeuron
from .dendrite import MPJRDDendrite
from .homeostasis import HomeostasisController
from .neuromodulation import Neuromodulator
from .accumulator import StatisticsAccumulator, AccumulatedStats
from ..utils.types import LearningMode
from ..utils.validation import validate_input, validate_device_consistency

# ✅ LOGGING
from ..utils.logging import get_logger

# Telemetria (import opcional)
try:
    from ..telemetry import TelemetryController, TelemetryConfig, forward_event, commit_event, sleep_event
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False


class MPJRDNeuron(BaseNeuron):
    """
    Neurônio MPJRD completo.
    
    ✅ OTIMIZADO:
        - Forward pass eficiente
        - Suporte a learning_rate_multiplier
        - Usa activity_threshold da config
        - ✅ LOGGING integrado
        - ✅ Validação de devices
    """

    def __init__(self, cfg: MPJRDConfig, enable_telemetry: bool = False,
                 telemetry_profile: str = "off", name: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        
        # ✅ LOGGER específico para este neurônio
        self.logger = get_logger(f"pyfolds.neuron.{name or id(self)}")
        self.logger.debug(f"🔧 Inicializando neurônio com config: {cfg}")
        self.logger.debug(f"   📊 D={cfg.n_dendrites}, S={cfg.n_synapses_per_dendrite}")

        # ===== COMPONENTES =====
        self.dendrites = nn.ModuleList([
            MPJRDDendrite(cfg, i) for i in range(cfg.n_dendrites)
        ])
        self.logger.debug(f"   ✅ {len(self.dendrites)} dendritos criados")
        
        self.homeostasis = HomeostasisController(cfg)
        self.neuromodulator = Neuromodulator(cfg)
        
        # Registra callbacks de homeostase
        self.homeostasis.on_stable(self._on_homeostasis_stable)
        self.homeostasis.on_unstable(self._on_homeostasis_unstable)
        
        # Sistema de acumulação unificado
        self.stats_acc = StatisticsAccumulator(
            cfg.n_dendrites, cfg.n_synapses_per_dendrite, cfg.eps
        )
        self.logger.debug(f"   ✅ Accumulator criado (track_extra={self.stats_acc.track_extra})")

        # ===== ESTADO =====
        self.mode = LearningMode.ONLINE
        self.register_buffer("mode_switches", torch.tensor(0))
        self.register_buffer("sleep_count", torch.tensor(0))
        self.logger.debug(f"   ✅ Modo inicial: {self.mode.value}")
        
        # ===== TELEMETRIA =====
        self.register_buffer("step_id", torch.tensor(0, dtype=torch.int64))
        if TELEMETRY_AVAILABLE and enable_telemetry:
            from ..telemetry import TelemetryConfig, TelemetryController
            telem_cfg = TelemetryConfig(
                profile=telemetry_profile if telemetry_profile in ["off", "light", "heavy"] else "off",
                sample_every=50 if telemetry_profile == "light" else 1,
                memory_capacity=256 if telemetry_profile == "light" else 2000
            )
            self.telemetry = TelemetryController(telem_cfg)
            self.logger.debug(f"   ✅ Telemetria ativada (profile={telemetry_profile})")
        else:
            self.telemetry = None
            self.logger.debug("   ⏭️ Telemetria desativada")
        
        # Valida devices após inicialização
        self._validate_internal_devices()
        
        self.logger.info(f"✅ Neurônio {name or id(self)} inicializado com sucesso")

    def _validate_internal_devices(self) -> None:
        """Valida consistência de devices internos."""
        devices = set()
        
        # Device do theta
        devices.add(self.theta.device)
        
        # Devices das dendrites e sinapses
        for dend in self.dendrites:
            for syn in dend.synapses:
                devices.add(syn.N.device)
                devices.add(syn.I.device)
        
        if len(devices) > 1:
            self.logger.error(f"❌ Devices inconsistentes: {devices}")
            raise RuntimeError(
                f"Componentes do neurônio em devices diferentes: {devices}. "
                "Todos devem estar no mesmo device."
            )
        
        self.logger.debug(f"✅ Devices consistentes: {devices.pop()}")

    def _on_homeostasis_stable(self, controller: HomeostasisController) -> None:
        """Callback quando homeostase se torna estável."""
        self.logger.info(
            f"✅ Homeostase estável! θ={controller.theta.item():.3f}, "
            f"erro={controller.homeostasis_error.item():.3f}"
        )
        # Opcional: reduz learning rate ou muda modo
        if self.mode == LearningMode.ONLINE:
            self.logger.debug("📉 Considerando reduzir learning rate...")

    def _on_homeostasis_unstable(self, controller: HomeostasisController) -> None:
        """Callback quando homeostase se torna instável."""
        self.logger.warning(
            f"⚠️ Homeostase instável! θ={controller.theta.item():.3f}, "
            f"erro={controller.homeostasis_error.item():.3f}"
        )

    # ========== PROPRIEDADES AGREGADAS ==========

    @property
    def N(self) -> torch.Tensor:
        """Filamentos [dendrites, synapses]."""
        return torch.stack([d.N for d in self.dendrites])

    @property
    def I(self) -> torch.Tensor:
        """Potenciais internos [dendrites, synapses]."""
        return torch.stack([d.I for d in self.dendrites])

    @property
    def W(self) -> torch.Tensor:
        """Pesos [dendrites, synapses]."""
        return torch.stack([d.W for d in self.dendrites])

    @property
    def protection(self) -> torch.Tensor:
        """Flags de proteção [dendrites, synapses]."""
        prot_list = []
        for dend in self.dendrites:
            dend_prot = torch.stack([syn.protection.squeeze() for syn in dend.synapses])
            prot_list.append(dend_prot)
        return torch.stack(prot_list)

    @property
    def theta(self) -> torch.Tensor:
        return self.homeostasis.theta

    @property
    def r_hat(self) -> torch.Tensor:
        return self.homeostasis.r_hat

    # ========== CONTROLE DE MODO ==========

    def set_mode(self, mode: LearningMode) -> None:
        """Define modo de aprendizado."""
        if mode != self.mode:
            old_mode = self.mode.value
            self.mode = mode
            self.mode_switches.add_(1)
            self.logger.info(f"🔄 Modo alterado: {old_mode} → {mode.value}")
            
            # Ações específicas por modo
            if mode == LearningMode.SLEEP:
                self.stats_acc.reset()  # Limpa pendências antes de dormir

    # ========== NEUROMODULAÇÃO ENDÓGENA ==========

    @torch.no_grad()
    def _compute_R_endogenous(self, current_rate: float, saturation_ratio: float) -> float:
        """Computa sinal neuromodulador interno."""
        cfg = self.cfg

        if cfg.neuromod_mode == "external":
            return 0.0

        if cfg.neuromod_mode == "capacity":
            cap = 1.0 - saturation_ratio
            rate_pen = abs(current_rate - cfg.target_spike_rate)
            R = cfg.cap_bias + cfg.cap_k_sat * cap - cfg.cap_k_rate * rate_pen
            return float(max(-1.0, min(1.0, R)))

        if cfg.neuromod_mode == "surprise":
            surprise = abs(current_rate - float(self.r_hat.item()))
            R = cfg.sup_bias + cfg.sup_k * surprise
            return float(max(-1.0, min(1.0, R)))

        return 0.0

    # ========== FORWARD PASS ==========

    def _validate_input_device(self, x: torch.Tensor) -> None:
        """Valida se input está no device correto."""
        if x.device != self.theta.device:
            raise RuntimeError(
                f"Input device {x.device} != neuron device {self.theta.device}. "
                "Use .to(device) no input ou mova o neurônio."
            )

    @torch.no_grad()
    def _apply_online_plasticity(
        self,
        x: torch.Tensor,
        post_rate: float,
        R_tensor: torch.Tensor,
        dt: float,
        mode: LearningMode,
    ) -> None:
        """Aplica regra local imediatamente (modo ONLINE sem defer)."""
        cfg = self.cfg
        post_rate_t = torch.tensor([max(0.0, min(1.0, post_rate))], device=self.theta.device)

        for d_idx, dend in enumerate(self.dendrites):
            # Filtra sinapses ativas
            active_mask = (x[:, d_idx, :] > cfg.activity_threshold).float()
            active_count = active_mask.sum(dim=0).clamp_min(1.0)
            pre_rate = (x[:, d_idx, :] * active_mask).sum(dim=0) / active_count
            pre_rate = pre_rate.clamp(0.0, 1.0)

            dend.update_synapses_rate_based(
                pre_rate=pre_rate,
                post_rate=post_rate_t,
                R=R_tensor,
                dt=dt,
                mode=mode,
            )

    @validate_input(
        expected_ndim=3,
        expected_shape_fn=lambda self: (self.cfg.n_dendrites, self.cfg.n_synapses_per_dendrite),
    )
    def forward(self, x: torch.Tensor, reward: Optional[float] = None,
                mode: Optional[LearningMode] = None,
                collect_stats: bool = True,
                dt: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Forward pass do neurônio.
        
        Args:
            x: Tensor de entrada [batch, dendrites, synapses]
            reward: Sinal de recompensa externo
            mode: Modo de aprendizado (sobrescreve o atual)
            collect_stats: Se deve coletar estatísticas
            dt: Passo de tempo (ms)
        
        Returns:
            Dict com spikes, potenciais e estatísticas
        """
        effective_mode = mode if mode is not None else self.mode
        
        # Valida device
        self._validate_input_device(x)
        
        device = self.theta.device
        B, D, _ = x.shape

        # ===== 1. INTEGRAÇÃO DENDRÍTICA =====
        v_dend = torch.zeros(B, D, device=device)
        for d_idx, dend in enumerate(self.dendrites):
            v_dend[:, d_idx] = dend(x[:, d_idx, :])

        # ===== 2. WTA (Winner-Take-All) =====
        max_val, max_idx = v_dend.max(dim=1, keepdim=True)
        gated = torch.zeros_like(v_dend)
        gated.scatter_(1, max_idx, v_dend.gather(1, max_idx))

        # ===== 3. POTENCIAL SOMÁTICO =====
        u = gated.sum(dim=1)

        # ===== 4. DISPARO =====
        spikes = (u >= self.theta).float()

        # ===== 5. ESTATÍSTICAS =====
        spike_rate = spikes.mean().item()
        saturation_ratio = (self.N == self.cfg.n_max).float().mean().item()

        # ===== 6. HOMEOSTASE =====
        if effective_mode != LearningMode.INFERENCE and collect_stats:
            self.homeostasis.update(spike_rate)

        # ===== 7. NEUROMODULAÇÃO =====
        if self.cfg.neuromod_mode == "external":
            R_val = float(reward) if reward is not None else 0.0
            R_val = max(-1.0, min(1.0, R_val))
        else:
            R_val = self._compute_R_endogenous(spike_rate, saturation_ratio)
        R_tensor = torch.tensor([R_val], device=device)

        # ===== 8. ACUMULAÇÃO (BATCH MODE) =====
        if collect_stats and effective_mode == LearningMode.BATCH and self.cfg.defer_updates:
            self.stats_acc.accumulate(x.detach(), gated.detach(), spikes.detach())

        # ===== 8b. ATUALIZAÇÃO IMEDIATA (ONLINE) =====
        if (
            collect_stats
            and effective_mode == LearningMode.ONLINE
            and self.cfg.plastic
            and not self.cfg.defer_updates
        ):
            self._apply_online_plasticity(
                x=x.detach(),
                post_rate=spike_rate,
                R_tensor=R_tensor,
                dt=dt,
                mode=effective_mode,
            )

        # ===== 9. TELEMETRIA =====
        self.step_id.add_(1)
        if self.telemetry is not None and self.telemetry.enabled():
            sid = int(self.step_id.item())
            self.telemetry.emit(forward_event(
                step_id=sid,
                mode=self.mode.value,
                spike_rate=spike_rate,
                theta=float(self.theta.item()),
                r_hat=float(self.r_hat.item()),
                saturation_ratio=saturation_ratio,
                R=float(R_tensor.item()),
                N_mean=float(self.N.float().mean().item()),
                I_mean=float(self.I.float().mean().item()),
                W_mean=float(self.W.float().mean().item()),
            ))

        # ===== 10. LOGGING =====
        self.logger.trace(
            f"Forward: batch={B}, spikes={spike_rate:.3f}, "
            f"θ={self.theta.item():.3f}, sat={saturation_ratio:.1%}"
        )
        
        if spike_rate < self.cfg.dead_neuron_threshold:
            self.logger.warning(
                f"⚠️ Neurônio hipoativo! rate={spike_rate:.3f} "
                f"(threshold={self.cfg.dead_neuron_threshold})"
            )
        
        if saturation_ratio > 0.5:
            self.logger.info(f"📊 Saturação alta: {saturation_ratio:.1%}")
        
        if R_val > 0.8:
            self.logger.debug(f"🎯 Neuromodulação alta: R={R_val:.2f}")

        # ===== 11. RETORNO =====
        return {
            "spikes": spikes,
            "u": u,
            "v_dend": v_dend,
            "gated": gated,
            "theta": self.theta.clone(),
            "r_hat": self.r_hat.clone(),
            "spike_rate": torch.tensor(spike_rate, device=device),
            "saturation_ratio": torch.tensor(saturation_ratio, device=device),
            "R": R_tensor,
            "N_mean": self.N.float().mean().to(device),
            "W_mean": self.W.float().mean().to(device),
            "I_mean": self.I.float().mean().to(device),
            "mode": self.mode.value,
        }

    def step(
        self,
        x: torch.Tensor,
        reward: Optional[float] = None,
        dt: float = 1.0,
        mode: Optional[LearningMode] = None,
        collect_stats: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """API explícita de passo temporal (compatível com README)."""
        return self.forward(
            x=x,
            reward=reward,
            mode=mode,
            collect_stats=collect_stats,
            dt=dt,
        )

    # ========== APLICAÇÃO DE PLASTICIDADE (BATCH) ==========

    @torch.no_grad()
    def apply_plasticity(self, dt: float = 1.0, reward: Optional[float] = None) -> None:
        """
        Aplica plasticidade acumulada (deferred updates).
        
        ✅ CORRIGIDO: Passa mode para as sinapses
        """
        if not self.cfg.plastic:
            self.logger.debug("⏭️ Plasticidade desabilitada (cfg.plastic=False)")
            return

        stats = self.stats_acc.get_averages()

        if not stats.is_valid():
            self.logger.debug("⏭️ Sem dados acumulados para plasticidade")
            return

        self.logger.debug(
            f"🔄 Aplicando plasticidade: dt={dt}, "
            f"reward={reward}, amostras={stats.total_samples}"
        )

        x_mean = stats.x_mean
        gated_mean = stats.gated_mean
        post_rate_float = float(stats.post_rate)

        cfg = self.cfg
        device = self.theta.device

        post_rate_float = max(0.0, min(1.0, post_rate_float))
        post_rate_t = torch.tensor([post_rate_float], device=device)

        saturation_ratio = (self.N == cfg.n_max).float().mean().item()

        if cfg.neuromod_mode == "external":
            R = float(reward) if reward is not None else 0.0
            R = max(-1.0, min(1.0, R))
        else:
            R = self._compute_R_endogenous(post_rate_float, saturation_ratio)
        R_t = torch.tensor([R], device=device)

        # Atualiza cada dendrito
        for d_idx, dend in enumerate(self.dendrites):
            # Taxas pré: média sobre sinapses ativas
            active_mask = (x_mean[d_idx] > cfg.activity_threshold).float()
            n_active = active_mask.sum().clamp_min(1.0)
            pre_rate = (x_mean[d_idx] * active_mask) / n_active
            pre_rate = pre_rate.clamp(0.0, 1.0)

            dend.update_synapses_rate_based(
                pre_rate=pre_rate,
                post_rate=post_rate_t,
                R=R_t,
                dt=dt,
                mode=self.mode
            )

        self.stats_acc.reset()
        self.logger.debug(f"✅ Plasticidade aplicada, R={R:.3f}, post_rate={post_rate_float:.3f}")

        if self.telemetry is not None and self.telemetry.enabled():
            sid = int(self.step_id.item())
            self.telemetry.emit(commit_event(
                step_id=sid,
                mode=self.mode.value,
                committed=True,
                post_rate=post_rate_float,
                R=R,
            ))

    # ========== CICLO DE SONO ==========

    @torch.no_grad()
    def sleep(self, duration: float = 60.0) -> None:
        """Ciclo de sono: consolida sinapses."""
        self.logger.info(f"💤 Iniciando sono por {duration}ms")
        self.sleep_count.add_(1)
        
        for i, dend in enumerate(self.dendrites):
            self.logger.debug(f"   Dendrito {i}: consolidando...")
            dend.consolidate(dt=duration)
        
        self.logger.info(f"✅ Sono concluído")

        if self.telemetry is not None and self.telemetry.enabled():
            sid = int(self.step_id.item())
            self.telemetry.emit(sleep_event(
                step_id=sid,
                mode=self.mode.value,
                duration=float(duration),
            ))

    # ========== UTILITÁRIOS ==========

    def to(self, device: torch.device) -> 'MPJRDNeuron':
        """Move neurônio para device e valida consistência."""
        super().to(device)
        self._validate_internal_devices()
        self.logger.debug(f"📦 Neurônio movido para {device}")
        return self

    # ========== MÉTRICAS ==========

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas consolidadas do neurônio."""
        N_flat = self.N.float().flatten()
        I_flat = self.I.flatten()

        if len(N_flat) > 0:
            percentiles = torch.quantile(N_flat, torch.tensor([0.25, 0.5, 0.75]))
        else:
            percentiles = torch.tensor([0.0, 0.0, 0.0])
            
        metrics = {
            'type': 'MPJRDNeuron',
            'id': id(self),
            'theta': self.theta.item(),
            'r_hat': self.r_hat.item(),
            'step_count': self.homeostasis.step_count.item(),
            'N_mean': N_flat.mean().item(),
            'N_std': N_flat.std().item(),
            'N_min': N_flat.min().item(),
            'N_max': N_flat.max().item(),
            'N_25p': percentiles[0].item(),
            'N_median': percentiles[1].item(),
            'N_75p': percentiles[2].item(),
            'I_mean': I_flat.mean().item(),
            'I_std': I_flat.std().item(),
            'I_min': I_flat.min().item(),
            'I_max': I_flat.max().item(),
            'saturation_ratio': (self.N == self.cfg.n_max).float().mean().item(),
            'protection_ratio': self.protection.float().mean().item(),
            'total_synapses': self.N.numel(),
            'total_dendrites': len(self.dendrites),
            'mode': self.mode.value,
            'mode_switches': self.mode_switches.item(),
            'sleep_count': self.sleep_count.item(),
            'has_pending_updates': self.stats_acc.has_data,
            'pending_count': self.stats_acc.acc_count.item() if self.stats_acc.has_data else 0,
            'device': str(self.theta.device),
            'homeostasis_stable': self.homeostasis.is_stable(),
        }
        
        self.logger.debug(f"📊 Métricas coletadas: N_mean={metrics['N_mean']:.1f}, θ={metrics['theta']:.2f}")
        return metrics

    def extra_repr(self) -> str:
        return (f"mode={self.mode.value}, D={self.cfg.n_dendrites}, "
                f"S={self.cfg.n_synapses_per_dendrite}, θ={self.theta.item():.2f}, "
                f"device={self.theta.device}")