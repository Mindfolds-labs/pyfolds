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
from typing import Optional, Dict, Any, Union
from threading import Lock
from queue import Queue, Empty, Full
from dataclasses import fields
from .config import MPJRDConfig
from .base import BaseNeuron
from .dendrite import MPJRDDendrite
from .homeostasis import HomeostasisController
from .neuromodulation import Neuromodulator
from .accumulator import StatisticsAccumulator
from .dendrite_integration import DendriticIntegration
from .cognitive_controller import (
    NetworkOrientationController,
    NetworkState,
    NeuromodulatoryState,
    OrientationPolicy,
    state_transition_event,
)
from .scientific_contract import (
    ContractEnforcer,
    ScientificContract,
    ScientificStage,
)
from ..utils.types import LearningMode, normalize_learning_mode
from ..utils.validation import validate_input

# ✅ LOGGING
from ..utils.logging import get_logger

# Telemetria (import opcional)
try:
    from ..telemetry import forward_event, commit_event, sleep_event

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
        - ✅ Thread-safe telemetry
    """

    def __init__(
        self,
        cfg: MPJRDConfig,
        enable_telemetry: bool = False,
        telemetry_profile: str = "off",
        audit_mode: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self._base_i_eta = float(cfg.i_eta)
        self._active_i_eta = float(cfg.i_eta)
        self.audit_mode = str(audit_mode or getattr(cfg, "audit_mode", "off"))
        self._contract_enforcer = ContractEnforcer(
            level=str(getattr(cfg, "contract_enforcement", "warn"))
        )

        # ✅ LOGGER específico para este neurônio
        self.logger = get_logger(f"pyfolds.neuron.{name or id(self)}")
        self.logger.debug(f"🔧 Inicializando neurônio com config: {cfg}")
        self.logger.debug(f"   📊 D={cfg.n_dendrites}, S={cfg.n_synapses_per_dendrite}")

        # ===== COMPONENTES =====
        self.dendrites = nn.ModuleList(
            [MPJRDDendrite(cfg, i) for i in range(cfg.n_dendrites)]
        )
        self.logger.debug(f"   ✅ {len(self.dendrites)} dendritos criados")

        self.homeostasis = HomeostasisController(cfg)
        self.neuromodulator = Neuromodulator(cfg)

        # Registra callbacks de homeostase
        self.homeostasis.on_stable(self._on_homeostasis_stable)
        self.homeostasis.on_unstable(self._on_homeostasis_unstable)

        # Sistema de acumulação unificado
        self.stats_acc = StatisticsAccumulator(
            cfg.n_dendrites,
            cfg.n_synapses_per_dendrite,
            cfg.eps,
            mode=cfg.stats_accumulator_mode,
            activity_threshold=cfg.activity_threshold,
            sparse_min_activity_ratio=cfg.sparse_min_activity_ratio,
            scientific_debug_stats=cfg.scientific_debug_stats,
            enable_profiling=cfg.enable_accumulator_profiling,
        )
        self.logger.debug(
            f"   ✅ Accumulator criado (track_extra={self.stats_acc.track_extra})"
        )

        # Integração dendrítica (substitui WTA hard)
        self.dendrite_integration = DendriticIntegration(cfg)
        self.logger.debug(
            "   ✅ DendriticIntegration: mode=%s, gain=%.1f, θ_ratio=%.2f, shunting_eps=%.2f",
            cfg.dendrite_integration_mode,
            cfg.dendrite_gain,
            cfg.theta_dend_ratio,
            cfg.shunting_eps,
        )

        # Cache de pesos dendríticos consolidados para evitar torch.stack por passo
        self._cached_consolidated_weights: Optional[torch.Tensor] = None
        self._weight_cache_dirty = True
        self._weight_cache_enabled = bool(cfg.enable_weight_cache)
        self._weight_cache_rebuilds = 0

        # ===== ESTADO =====
        self.mode = LearningMode.ONLINE
        self.network_state = NetworkState.ACTIVE
        self.orientation_controller = NetworkOrientationController(
            base_eta=self._base_i_eta,
            replay_interval_steps=int(getattr(cfg, "replay_interval_steps", 32)),
        )
        self._latest_policy = OrientationPolicy(
            current_mode=self.network_state,
            effective_eta=self._base_i_eta,
            effective_attention_gain=1.0,
            effective_competition_gain=1.0,
            effective_replay_priority=0.0,
            effective_consolidation_rate=1.0,
            effective_decay_rate=1.0,
            sensory_excitability=1.0,
        )
        self._latest_neuromodulatory_state = NeuromodulatoryState(0.5, 0.5, 0.5, 0.5)
        self.register_buffer("mode_switches", torch.tensor(0), persistent=True)
        self.register_buffer("sleep_count", torch.tensor(0), persistent=True)
        self.register_buffer(
            "connectivity_mask",
            torch.ones(
                (cfg.n_dendrites, cfg.n_synapses_per_dendrite),
                dtype=torch.float32,
                device=self.theta.device,
            ),
            persistent=True,
        )
        self.register_buffer(
            "pruning_mask",
            torch.ones_like(self.connectivity_mask),
            persistent=True,
        )
        self.register_buffer(
            "phase_activity_hist",
            torch.zeros(int(cfg.circadian_phase_bins), dtype=torch.float32, device=self.theta.device),
            persistent=False,
        )
        self.register_buffer(
            "_runtime_resonance_cache",
            torch.zeros(int(cfg.n_dendrites), dtype=torch.float32, device=self.theta.device),
            persistent=False,
        )
        self.logger.debug(f"   ✅ Modo inicial: {self.mode.value}")

        # ===== THREAD SAFETY PARA TELEMETRIA =====
        self._telemetry_lock = Lock()

        # ===== FILA LOCK-FREE DE INJEÇÕES RUNTIME (MindControl) =====
        self._runtime_injections: Queue[tuple[str, Any]] = Queue(
            maxsize=max(1, int(getattr(cfg, "runtime_queue_maxsize", 2048)))
        )

        # ===== TELEMETRIA =====
        self.register_buffer("step_id", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("global_time_ms", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("circadian_phase", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("circadian_encoding_gate", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("circadian_consolidation_gate", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("circadian_attention_gate", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("_ema_reward", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_ema_novelty", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_ema_spike_rate", torch.tensor(0.0, dtype=torch.float32))
        self._audit_events: list[dict[str, Any]] = []
        trace_capacity = int(getattr(cfg, "audit_trace_capacity", 512))
        self.register_buffer(
            "_trace_winner_idx",
            torch.full((trace_capacity,), -1, dtype=torch.int64, device=self.theta.device),
            persistent=False,
        )
        self.register_buffer(
            "_trace_signal",
            torch.zeros(trace_capacity, dtype=self.theta.dtype, device=self.theta.device),
            persistent=False,
        )
        self.register_buffer(
            "_trace_ptr", torch.tensor(0, dtype=torch.int64, device=self.theta.device), persistent=False
        )
        self.register_buffer(
            "_theta_cap_buf",
            torch.zeros(1, dtype=self.theta.dtype, device=self.theta.device),
            persistent=False,
        )
        self.register_buffer(
            "gate_activity_mean",
            torch.tensor(0.5, dtype=self.theta.dtype, device=self.theta.device),
            persistent=False,
        )
        self.register_buffer(
            "gate_logit_mean",
            torch.tensor(0.0, dtype=self.theta.dtype, device=self.theta.device),
            persistent=False,
        )
        self.register_buffer(
            "gate_std",
            torch.tensor(0.0, dtype=self.theta.dtype, device=self.theta.device),
            persistent=False,
        )
        self.register_buffer(
            "gate_logit_std",
            torch.tensor(0.0, dtype=self.theta.dtype, device=self.theta.device),
            persistent=False,
        )
        self.register_buffer(
            "u_eff_mean",
            torch.tensor(0.0, dtype=self.theta.dtype, device=self.theta.device),
            persistent=False,
        )
        self.register_buffer(
            "u_eff_std",
            torch.tensor(0.0, dtype=self.theta.dtype, device=self.theta.device),
            persistent=False,
        )
        if TELEMETRY_AVAILABLE and enable_telemetry:
            from ..telemetry import (
                TelemetryConfig,
                TelemetryController,
                TelemetryProfile,
            )

            profile_value = telemetry_profile
            if isinstance(profile_value, str) and profile_value == "full":
                profile_value = "heavy"
            profile_enum = (
                TelemetryProfile(profile_value)
                if isinstance(profile_value, str)
                else profile_value
            )
            is_light = profile_enum == TelemetryProfile.LIGHT
            telem_cfg = TelemetryConfig(
                profile=profile_enum,
                sample_every=50 if is_light else 1,
                memory_capacity=256 if is_light else 2000,
            )
            self.telemetry = TelemetryController(telem_cfg)
            self.logger.debug(
                f"   ✅ Telemetria ativada (profile={profile_enum.value})"
            )
        else:
            self.telemetry = None
            self.logger.debug("   ⏭️ Telemetria desativada")

        self._gradient_hook_handles = []
        self._install_gradient_health_monitor()

        self.logger.info(f"✅ Neurônio {name or id(self)} inicializado com sucesso")

        # Valida devices após inicialização completa (todos os buffers registrados)
        self._validate_internal_devices()

    def _install_gradient_health_monitor(self) -> None:
        """Instala saneamento de gradientes em nível de parâmetro."""
        for param in self.parameters():
            if not param.requires_grad:
                continue
            handle = param.register_hook(self._sanitize_gradient)
            self._gradient_hook_handles.append(handle)

    def _sanitize_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Gatekeeper contra gradientes corrompidos por falhas de hardware."""
        if torch.isfinite(grad).all():
            return grad

        self.logger.error("Gradiente inválido detectado; substituindo por zeros.")
        return torch.zeros_like(grad)

    def _validate_internal_devices(self) -> None:
        """Valida consistência de devices internos após inicialização completa."""
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

        expected_device = self.theta.device
        step_id_buf = getattr(self, "step_id", None)
        if step_id_buf is not None and step_id_buf.device != expected_device:
            raise RuntimeError(
                f"step_id device {step_id_buf.device} != theta device {expected_device}"
            )

        self.logger.debug(f"✅ Devices consistentes: {devices.pop()}")

    def _on_homeostasis_stable(self, controller: HomeostasisController) -> None:
        """Callback quando homeostase se torna estável."""
        self.logger.info(
            f"✅ Homeostase estável! θ={controller.theta.item():.3f}, "
            f"erro={controller.homeostasis_error.item():.3f}"
        )

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
    def L(self) -> torch.Tensor:
        """Níveis discretos de peso [dendrites, synapses]."""
        return torch.stack([d.L for d in self.dendrites])

    @property
    def I(self) -> torch.Tensor:  # noqa: E743
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

    def _compute_saturation_ratio(self) -> float:
        """Calcula taxa de saturação conforme modo de quantização de peso."""
        if self.cfg.weight_quantization == "uniformW":
            return (self.L == (self.cfg.n_levels - 1)).float().mean().item()
        return (self.N == self.cfg.n_max).float().mean().item()

    @torch.no_grad()
    def queue_runtime_injection(self, name: str, value: Any) -> None:
        """Enfileira mutação de parâmetro para aplicação assíncrona no próximo step."""
        try:
            self._runtime_injections.put_nowait((name, value))
        except Full:
            self.logger.warning(
                "Fila de injeções runtime cheia; descartando mutação %s", name
            )

    @torch.no_grad()
    def _refresh_config_references(self) -> None:
        """Propaga nova configuração para submódulos que cacheiam ``cfg``."""
        self.homeostasis.cfg = self.cfg
        self.neuromodulator.cfg = self.cfg
        self.stats_acc.cfg = (
            self.cfg if hasattr(self.stats_acc, "cfg") else getattr(self, "cfg")
        )
        self.dendrite_integration.cfg = self.cfg
        self.stats_acc.mode = self.cfg.stats_accumulator_mode
        self.stats_acc.activity_threshold = self.cfg.activity_threshold
        self.stats_acc.sparse_min_activity_ratio = self.cfg.sparse_min_activity_ratio
        self.stats_acc.scientific_debug_stats = self.cfg.scientific_debug_stats
        self.stats_acc.enable_profiling = self.cfg.enable_accumulator_profiling
        self._weight_cache_enabled = bool(self.cfg.enable_weight_cache)
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "cfg"):
                module.cfg = self.cfg

    @torch.no_grad()
    def _apply_runtime_injections(self) -> int:
        """Aplica injeções pendentes sem bloquear o caminho crítico do forward."""
        applied = 0
        allowed_fields = {f.name for f in fields(MPJRDConfig)}

        while True:
            try:
                name, value = self._runtime_injections.get_nowait()
            except Empty:
                break

            if isinstance(value, torch.Tensor):
                value = value.to(self.theta.device)

            if name == "theta":
                with torch.no_grad():
                    tensor_value = torch.as_tensor(
                        value, dtype=self.theta.dtype, device=self.theta.device
                    )
                    if tensor_value.numel() == 1:
                        tensor_value = tensor_value.reshape_as(self.theta)
                    self.theta.copy_(tensor_value)
                applied += 1
                continue

            if name in {"r_hat", "target_spike_rate"}:
                if name == "r_hat":
                    tensor_value = torch.as_tensor(
                        value, dtype=self.r_hat.dtype, device=self.r_hat.device
                    )
                    if tensor_value.numel() == 1:
                        tensor_value = tensor_value.reshape_as(self.r_hat)
                    self.r_hat.copy_(tensor_value)
                    applied += 1
                    continue

                cfg_field = self.cfg.resolve_runtime_alias(name)
                if cfg_field in allowed_fields:
                    self.cfg = self.cfg.with_runtime_update(**{cfg_field: float(value)})
                    if cfg_field == "i_eta":
                        self._base_i_eta = float(self.cfg.i_eta)
                    self._refresh_config_references()
                    applied += 1
                continue

            cfg_field = self.cfg.resolve_runtime_alias(name)
            if cfg_field in allowed_fields:
                casted = (
                    float(value)
                    if isinstance(value, (int, float, torch.Tensor))
                    else value
                )
                self.cfg = self.cfg.with_runtime_update(**{cfg_field: casted})
                if cfg_field == "i_eta":
                    self._base_i_eta = float(self.cfg.i_eta)
                self._refresh_config_references()
                applied += 1

        return applied

    def _safe_step_id(self) -> int:
        """Retorna step_id de forma segura mesmo sem buffer registrado."""
        step_id_buf = getattr(self, "step_id", None)
        if step_id_buf is None:
            return 0
        return int(step_id_buf.item())

    @torch.no_grad()
    def _append_audit_trace(self, winner_idx: torch.Tensor, winner_signal: torch.Tensor) -> None:
        """Append minimal decision trace to on-device circular audit buffer.

        Parameters
        ----------
        winner_idx : torch.Tensor
            Winner dendrite index tensor with shape ``[B]``.
        winner_signal : torch.Tensor
            Winner dendritic signal tensor with shape ``[B]``.
        """
        if self.audit_mode == "off":
            return
        ptr = int(self._trace_ptr.item())
        self._trace_winner_idx[ptr] = winner_idx.reshape(-1)[0].to(torch.int64)
        self._trace_signal[ptr] = winner_signal.reshape(-1)[0].to(self.theta.dtype)
        self._trace_ptr.fill_((ptr + 1) % self._trace_winner_idx.numel())

    def _update_circadian_gates(self, dt: float) -> None:
        """Update circadian phase and continuous operation gates.

        Parameters
        ----------
        dt : float
            Time increment in milliseconds.
        """
        self.global_time_ms.add_(float(dt))
        if not bool(getattr(self.cfg, "circadian_enabled", False)):
            self.circadian_phase.fill_(0.0)
            self.circadian_encoding_gate.fill_(1.0)
            self.circadian_consolidation_gate.fill_(0.0)
            self.circadian_attention_gate.fill_(1.0)
            return

        cycle_ms = max(float(getattr(self.cfg, "circadian_cycle_hours", 12.0)) * 3600.0 * 1000.0, 1e-6)
        phase = float((self.global_time_ms / cycle_ms) * (2.0 * torch.pi))
        self.circadian_phase.fill_(phase)

        encoding_raw = 0.5 + 0.5 * torch.cos(torch.tensor(phase, dtype=torch.float32))
        consolidation_raw = 1.0 - encoding_raw
        attention_raw = 0.6 * encoding_raw + 0.4

        min_gate = float(getattr(self.cfg, "circadian_plasticity_min", 0.1))
        max_gate = float(getattr(self.cfg, "circadian_plasticity_max", 1.5))
        self.circadian_encoding_gate.fill_(min_gate + (max_gate - min_gate) * float(encoding_raw.item()))
        self.circadian_consolidation_gate.fill_(float(consolidation_raw.item()))
        self.circadian_attention_gate.fill_(float(attention_raw.item()))

    def _estimate_pending_eligibility_mass(self) -> float:
        """Estimate current eligibility mass to drive consolidation decisions."""
        total_mass = 0.0
        for dend in self.dendrites:
            syn_batch = getattr(dend, "synapse_batch", None)
            if syn_batch is not None:
                total_mass += float(syn_batch.eligibility.abs().mean().item())
                total_mass += float(syn_batch.stdp_eligibility.abs().mean().item())
                continue
            for syn in dend.synapses:
                total_mass += float(syn.eligibility.abs().mean().item())
                total_mass += float(syn.stdp_eligibility.abs().mean().item())
        return total_mass / max(len(self.dendrites), 1)

    def update_network_state(
        self,
        reward_signal: float,
        novelty_estimate: float,
        surprise_estimate: float,
        saturation_ratio: float,
        stability_ratio: float,
    ) -> None:
        """Update high-level operating state from circadian and neuromodulatory context."""
        neuromod_state = self.orientation_controller.compute_neuromodulators(
            reward_signal=reward_signal,
            novelty_estimate=novelty_estimate,
            surprise_estimate=surprise_estimate,
            stability_ratio=stability_ratio,
        )
        self._latest_neuromodulatory_state = neuromod_state
        pending = self._estimate_pending_eligibility_mass()
        old_state = self.network_state
        self.network_state = self.orientation_controller.determine_operational_mode(
            current_state=self.network_state,
            circadian_encoding_gate=float(self.circadian_encoding_gate.item()),
            circadian_consolidation_gate=float(self.circadian_consolidation_gate.item()),
            pending_eligibility_mass=pending,
            saturation_ratio=saturation_ratio,
            stability_ratio=stability_ratio,
            novelty_estimate=novelty_estimate,
            step_index=self._safe_step_id(),
        )
        self._latest_policy = self.orientation_controller.compose_policy(
            state=self.network_state,
            circadian_attention_gate=float(self.circadian_attention_gate.item()),
            circadian_encoding_gate=float(self.circadian_encoding_gate.item()),
            circadian_consolidation_gate=float(self.circadian_consolidation_gate.item()),
            neuromodulatory_state=neuromod_state,
        )
        if old_state != self.network_state and self.audit_mode == "full":
            self._audit_events.append(state_transition_event(old_state, self.network_state, self._safe_step_id()))

    @torch.no_grad()
    def run_sleep_cycle(self, duration: float = 60.0) -> None:
        """Run sleep consolidation with decay/noise pruning semantics."""
        for dend in self.dendrites:
            dend.consolidate(dt=duration * self._latest_policy.effective_consolidation_rate)
        self.invalidate_weight_cache()

    @torch.no_grad()
    def run_replay_cycle(self) -> None:
        """Run compressed replay by partially reactivating high-eligibility traces."""
        replay_scale = max(0.0, self._latest_policy.effective_replay_priority)
        if replay_scale <= 0.0:
            return
        for dend in self.dendrites:
            syn_batch = getattr(dend, "synapse_batch", None)
            if syn_batch is not None:
                syn_batch.eligibility.mul_(1.0 + 0.1 * replay_scale)
                continue
            for syn in dend.synapses:
                syn.eligibility.mul_(1.0 + 0.1 * replay_scale)

        if bool(getattr(self.cfg, "consolidate_pruning_after_replay", False)):
            self._consolidate_pruning_from_runtime()

    @torch.no_grad()
    def apply_homeostatic_recovery(self) -> None:
        """Apply recovery mode gains to stabilize firing and threshold drift."""
        if self.network_state != NetworkState.HOMEOSTATIC_RECOVERY:
            return
        self.homeostasis.theta.mul_(1.01).clamp_(self.cfg.theta_min, self.cfg.theta_max)

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
    def _compute_R_endogenous(
        self, current_rate: float, saturation_ratio: float
    ) -> float:
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
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input deve ser torch.Tensor, recebido {type(x)}")
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
        effective_eta = self._active_i_eta
        cfg_for_step = (
            cfg.with_runtime_update(i_eta=effective_eta)
            if effective_eta != float(cfg.i_eta)
            else None
        )
        post_rate_t = torch.tensor(
            [max(0.0, min(1.0, post_rate))], device=self.theta.device
        )

        for d_idx, dend in enumerate(self.dendrites):
            # Filtra sinapses ativas
            active_mask = (x[:, d_idx, :] > cfg.activity_threshold).float()
            active_count = active_mask.sum(dim=0).clamp_min(1.0)
            pre_rate = (x[:, d_idx, :] * active_mask).sum(dim=0) / active_count
            pre_rate = pre_rate.clamp(0.0, 1.0)

            if cfg_for_step is not None:
                dend.cfg = cfg_for_step

            dend.update_synapses_rate_based(
                pre_rate=pre_rate,
                post_rate=post_rate_t,
                R=R_tensor,
                dt=dt,
                mode=mode,
            )

            if cfg_for_step is not None:
                dend.cfg = cfg

    def invalidate_weight_cache(self) -> None:
        """Marca cache de pesos como sujo (chamar após mutações de peso)."""
        self._weight_cache_dirty = True

    def _get_consolidated_weights(self, device: torch.device) -> torch.Tensor:
        """Retorna pesos [D,S] com cache opcional e invalidação explícita."""
        if not self._weight_cache_enabled:
            return torch.stack([d.W for d in self.dendrites], dim=0).to(device)

        if self._weight_cache_dirty or self._cached_consolidated_weights is None:
            self._cached_consolidated_weights = torch.stack([d.W for d in self.dendrites], dim=0)
            self._weight_cache_rebuilds += 1
            self._weight_cache_dirty = False

        if self._cached_consolidated_weights.device != device:
            self._cached_consolidated_weights = self._cached_consolidated_weights.to(device)

        return self._cached_consolidated_weights

    def _compute_dendritic_potentials_vectorized(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dendritic membrane potentials using a vectorized kernel.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``[batch, dendrites, synapses]``.

        Returns
        -------
        torch.Tensor
            Dendritic potentials with shape ``[batch, dendrites]``.

        Examples
        --------
        >>> cfg = MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)
        >>> neuron = MPJRDNeuron(cfg)
        >>> x = torch.randn(8, 2, 4)
        >>> neuron._compute_dendritic_potentials_vectorized(x).shape
        torch.Size([8, 2])
        """
        weights = self._get_consolidated_weights(x.device)
        effective_mask = (self.connectivity_mask * self.pruning_mask).to(device=x.device, dtype=weights.dtype)
        effective_weights = weights * effective_mask
        return torch.einsum("bds,ds->bd", x, effective_weights)

    @validate_input(
        expected_ndim=3,
        expected_shape_fn=lambda self: (
            self.cfg.n_dendrites,
            self.cfg.n_synapses_per_dendrite,
        ),
    )
    def _local_gate_logit(self, gate_drive: torch.Tensor, eps: float) -> torch.Tensor:
        """Normaliza drive de gate apenas com estatísticas espaciais por amostra."""
        local_mu = gate_drive.mean(dim=1, keepdim=True)
        local_sigma = gate_drive.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        return (gate_drive - local_mu) / local_sigma

    def forward(
        self,
        x: torch.Tensor,
        reward: Optional[float] = None,
        mode: Optional[Union[LearningMode, str]] = None,
        collect_stats: bool = True,
        dt: float = 1.0,
        defer_homeostasis: bool = False,
    ) -> Dict[str, Any]:
        """Execute one neuron simulation step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``[batch, dendrites, synapses]``.
        reward : float | None, optional
            External reward signal for neuromodulation.
        mode : LearningMode | str | None, optional
            Learning mode override for the current step.
        collect_stats : bool, default=True
            Whether to update running statistics.
        dt : float, default=1.0
            Time delta in milliseconds.
        defer_homeostasis : bool, default=False
            Skip homeostasis update for this step.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing spikes, potentials and diagnostic statistics.

        Examples
        --------
        >>> cfg = MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)
        >>> neuron = MPJRDNeuron(cfg)
        >>> out = neuron(torch.zeros(3, 2, 4))
        >>> sorted(["spikes", "u", "v_dend"]) <= sorted(out.keys())
        True
        """
        normalized_mode = normalize_learning_mode(mode)
        effective_mode = normalized_mode if normalized_mode is not None else self.mode
        execution_order: list[ScientificStage] = []

        self._update_circadian_gates(dt=dt)
        self._apply_runtime_injections()
        self._refresh_pruning_mask()

        # Valida device
        self._validate_input_device(x)

        if x.dim() != 3:
            raise ValueError(
                "Input do neurônio deve ter 3 dimensões [batch, dendrites, synapses]"
            )
        if not x.is_floating_point():
            raise TypeError("Input do neurônio deve ser tensor de ponto flutuante")

        device = self.theta.device
        B, D, _ = x.shape
        self._set_refractory_state(False)

        # ===== 1. INTEGRAÇÃO DENDRÍTICA =====
        use_vectorized_dendrites = getattr(self.cfg, "use_vectorized_dendrites", True)
        if use_vectorized_dendrites:
            v_dend = self._compute_dendritic_potentials_vectorized(x)
        else:
            x_masked = x * (self.connectivity_mask * self.pruning_mask).to(device=x.device, dtype=x.dtype).unsqueeze(0)
            dendrite_outputs = [
                dend(x_masked[:, d_idx, :]) for d_idx, dend in enumerate(self.dendrites)
            ]
            v_dend = torch.stack(dendrite_outputs, dim=1)

        if not torch.isfinite(v_dend).all():
            self.logger.warning(
                "event=non_finite_dendritic_sum action=nan_to_num_clamp mode=%s",
                (
                    effective_mode.value
                    if hasattr(effective_mode, "value")
                    else str(effective_mode)
                ),
            )
            v_dend = torch.nan_to_num(v_dend, nan=0.0, posinf=1e6, neginf=-1e6)

        v_dend = v_dend * float(self._latest_policy.sensory_excitability)

        if getattr(self.cfg, "backprop_enabled", True) and hasattr(
            self, "dendrite_amplification"
        ):
            max_gain = getattr(self.cfg, "backprop_max_gain", 2.0)
            amp = (1.0 + self.dendrite_amplification.to(device)).unsqueeze(0)
            amp = amp.clamp(1.0, max_gain)
            v_dend = v_dend * amp

        # ===== 2-4. INTEGRAÇÃO DENDRÍTICA → POTENCIAL → DISPARO =====
        integration_mode = getattr(self.cfg, "dendrite_integration_mode", "wta_hard")

        # Ajuste local do limiar para manter gates próximos da região sensível da sigmoide no início.
        theta_eff = torch.clamp(self.theta, min=self.cfg.theta_min, max=self.cfg.theta_max)
        if integration_mode == "nmda_shunting":
            # Capacidade máxima do somático com NMDA+shunting (v_nmda <= 1):
            # u_max = D² / (shunting_eps + shunting_strength * D)
            d_float = float(self.cfg.n_dendrites)
            u_cap = (d_float * d_float) / (
                self.cfg.shunting_eps + self.cfg.shunting_strength * d_float
            )
            theta_cap = u_cap - 1e-3
            theta_cap_buf = self._theta_cap_buf.to(
                device=device, dtype=self.theta.dtype
            )
            theta_cap_buf.fill_(theta_cap)
            theta_eff = torch.minimum(theta_eff, theta_cap_buf)
            theta_max_eff = min(self.cfg.theta_max, theta_cap)
            if self.cfg.theta_min <= theta_max_eff:
                theta_eff = torch.clamp(
                    theta_eff, min=self.cfg.theta_min, max=theta_max_eff
                )
            else:
                theta_eff = torch.clamp(theta_eff, max=theta_max_eff)

            dend_out = self.dendrite_integration(v_dend, theta_eff)
            execution_order.extend(
                [
                    ScientificStage.LOCAL_NONLINEARITY,
                    ScientificStage.COMPETITION,
                    ScientificStage.GLOBAL_AGGREGATION,
                ]
            )
            u = dend_out.u
            gated = dend_out.v_nmda
            gate_logit = dend_out.gate_logit
            dend_contribution = dend_out.contribution
        elif integration_mode == "wta_soft":
            theta_cap = float(self.cfg.n_dendrites) - 1e-3
            theta_cap_buf = self._theta_cap_buf.to(
                device=device, dtype=self.theta.dtype
            )
            theta_cap_buf.fill_(theta_cap)
            theta_eff = torch.minimum(theta_eff, theta_cap_buf)
            theta_max_eff = min(self.cfg.theta_max, theta_cap)
            if self.cfg.theta_min <= theta_max_eff:
                theta_eff = torch.clamp(
                    theta_eff, min=self.cfg.theta_min, max=theta_max_eff
                )
            else:
                theta_eff = torch.clamp(theta_eff, max=theta_max_eff)

            gate_drive = v_dend - (theta_eff * 0.5)
            gate_logit = self._local_gate_logit(gate_drive, self.cfg.gate_local_norm_eps)
            gate_logit = self.cfg.gate_logit_scale * gate_logit
            gated = torch.sigmoid(gate_logit)
            execution_order.append(ScientificStage.LOCAL_NONLINEARITY)
            u = gated.sum(dim=1)
            execution_order.append(ScientificStage.GLOBAL_AGGREGATION)
            dend_contribution = None
        else:
            max_idx = v_dend.max(dim=1, keepdim=True)[1]
            gated = torch.zeros_like(v_dend)
            gated.scatter_(1, max_idx, v_dend.gather(1, max_idx))
            execution_order.extend(
                [ScientificStage.COMPETITION, ScientificStage.GLOBAL_AGGREGATION]
            )
            u = gated.sum(dim=1)
            gate_logit = v_dend
            dend_contribution = None

        self._contract_enforcer.validate(ScientificContract(stage_order=execution_order))

        u_raw = u
        spikes = (u >= theta_eff).float()
        winner_idx = v_dend.argmax(dim=1)
        winner_signal = v_dend.gather(1, winner_idx.unsqueeze(1)).squeeze(1)
        self._runtime_resonance_cache.copy_(v_dend.abs().mean(dim=0).to(self._runtime_resonance_cache.dtype))
        self.gate_activity_mean.lerp_(gated.mean().detach().to(self.gate_activity_mean.dtype), weight=0.05)
        self.gate_logit_mean.copy_(gate_logit.mean().detach().to(self.gate_logit_mean.dtype))
        self.gate_std.copy_(gated.std(unbiased=False).detach().to(self.gate_std.dtype))
        self.gate_logit_std.copy_(gate_logit.std(unbiased=False).detach().to(self.gate_logit_std.dtype))
        self.u_eff_mean.copy_(u.mean().detach().to(self.u_eff_mean.dtype))
        self.u_eff_std.copy_(u.std(unbiased=False).detach().to(self.u_eff_std.dtype))
        self._append_audit_trace(winner_idx, winner_signal)

        # ===== 5. ESTATÍSTICAS =====
        spike_rate = spikes.mean().item()
        saturation_ratio = (
            self._compute_saturation_ratio()
            if self.cfg.neuromod_mode == "capacity"
            else 0.0
        )
        novelty_estimate = float((v_dend.std(dim=1).mean() / (v_dend.abs().mean() + 1e-6)).clamp(0.0, 1.0).item())
        reward_signal = float(reward if reward is not None else 0.0)
        surprise_estimate = abs(spike_rate - float(self.r_hat.item()))
        stability_ratio = 1.0 / (1.0 + abs(float(self.homeostasis.homeostasis_error.item())))
        self._ema_reward.mul_(0.97).add_(0.03 * reward_signal)
        self._ema_novelty.mul_(0.97).add_(0.03 * novelty_estimate)
        self._ema_spike_rate.mul_(0.97).add_(0.03 * spike_rate)
        phase_bin = int((float(self.circadian_phase.item()) % 360.0) / max(1e-6, (360.0 / max(1, int(self.cfg.circadian_phase_bins)))))
        phase_bin = max(0, min(int(self.phase_activity_hist.numel()) - 1, phase_bin))
        self.phase_activity_hist[phase_bin].add_(float(spike_rate))
        self.update_network_state(
            reward_signal=float(self._ema_reward.item()),
            novelty_estimate=float(self._ema_novelty.item()),
            surprise_estimate=float(surprise_estimate),
            saturation_ratio=float(saturation_ratio),
            stability_ratio=float(stability_ratio),
        )
        self._active_i_eta = float(self._latest_policy.effective_eta)

        if self.network_state == NetworkState.SLEEP_CONSOLIDATION:
            self.run_sleep_cycle(duration=dt)
        elif self.network_state == NetworkState.MEMORY_REPLAY:
            self.run_replay_cycle()
        elif self.network_state == NetworkState.HOMEOSTATIC_RECOVERY:
            self.apply_homeostatic_recovery()

        # ===== 6. VALIDAÇÃO ANTES DE HOMEOSTASE =====
        if not (isinstance(spike_rate, float) and -0.1 <= spike_rate <= 1.1):
            self.logger.warning(
                "event=rate_out_of_range metric=spike_rate value=%.6f expected=[0,1] "
                "mode=%s integration_mode=%s action=clamp_to_valid_range",
                spike_rate,
                (
                    effective_mode.value
                    if hasattr(effective_mode, "value")
                    else str(effective_mode)
                ),
                integration_mode,
            )
            spike_rate = max(0.0, min(1.0, spike_rate))

        # ===== 7. HOMEOSTASE =====
        if (
            effective_mode != LearningMode.INFERENCE
            and collect_stats
            and not defer_homeostasis
        ):
            self.homeostasis.update(
                spike_rate,
                in_refractory_period=self.in_refractory_period,
            )

        # ===== 8. NEUROMODULAÇÃO =====
        if self.cfg.neuromod_mode == "external":
            R_val = float(reward) if reward is not None else 0.0
            R_val = max(-1.0, min(1.0, R_val))
        else:
            R_val = self._compute_R_endogenous(spike_rate, saturation_ratio)
        R_tensor = torch.tensor([R_val], device=device)

        # ===== 9. ACUMULAÇÃO (BATCH MODE) =====
        acc_telem = None
        if (
            collect_stats
            and effective_mode == LearningMode.BATCH
            and self.cfg.defer_updates
        ):
            self.stats_acc.accumulate(x.detach(), gated.detach(), spikes.detach())
            acc_telem = self.stats_acc.telemetry_snapshot
        else:
            acc_telem = None

        # ===== 10. ATUALIZAÇÃO IMEDIATA (ONLINE) =====
        if (
            collect_stats
            and effective_mode == LearningMode.ONLINE
            and self.cfg.plastic
        ):
            self._apply_online_plasticity(
                x=x.detach(),
                post_rate=spike_rate,
                R_tensor=R_tensor,
                dt=dt,
                mode=effective_mode,
            )

        # ===== 11. TELEMETRIA COM SINCRONIZAÇÃO =====
        emit_telemetry = (
            self.telemetry is not None
            and self.telemetry.enabled()
            and collect_stats
            and effective_mode != LearningMode.INFERENCE
        )

        with self._telemetry_lock:
            step_buf = getattr(self, "step_id", None)
            if step_buf is not None:
                step_buf.add_(1)
            sid = self._safe_step_id()

            if emit_telemetry and self.telemetry.should_emit(sid):
                try:
                    self.telemetry.emit(
                        forward_event(
                            step_id=sid,
                            mode=effective_mode.value,
                            spike_rate=spike_rate,
                            theta=float(self.theta.item()),
                            r_hat=float(self.r_hat.item()),
                            saturation_ratio=saturation_ratio,
                            R=float(R_tensor.item()),
                            N_mean=float(self.N.float().mean().item()),
                            I_mean=float(self.I.float().mean().item()),
                            W_mean=float(self.W.float().mean().item()),
                            integration_mode=integration_mode,
                            accumulator_time_ms=(acc_telem["accumulator_time_ms"] if acc_telem else 0.0),
                            accumulator_activity_ratio=(acc_telem["activity_ratio"] if acc_telem else 0.0),
                            accumulator_sparse_path_used=(acc_telem["sparse_path_used"] if acc_telem else False),
                            accumulator_dense_fallback_used=(acc_telem["dense_fallback_used"] if acc_telem else False),
                            accumulator_nonzero_sample_ratio=(acc_telem["nonzero_sample_ratio"] if acc_telem else 0.0),
                        )
                    )
                except Exception as exc:
                    self.logger.error(f"Falha ao emitir telemetria: {exc}")

        # ===== 12. LOGGING =====
        self.logger.trace(
            f"Forward [{integration_mode}]: batch={B}, spikes={spike_rate:.3f}, "
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

        # ===== 13. RETORNO =====
        out = {
            "spikes": spikes,
            "u": u,
            "u_raw": u_raw,
            "u_eff": u,
            "v_dend": v_dend,
            "gated": gated,
            "gate_logit": gate_logit,
            "gate_activity_mean": self.gate_activity_mean.clone(),
            "gate_mean": self.gate_activity_mean.clone(),
            "gate_std": self.gate_std.clone(),
            "gate_logit_mean": self.gate_logit_mean.clone(),
            "gate_logit_std": self.gate_logit_std.clone(),
            "u_eff_mean": self.u_eff_mean.clone(),
            "u_eff_std": self.u_eff_std.clone(),
            "theta": self.theta.clone(),
            "theta_eff": theta_eff.clone(),
            "r_hat": self.r_hat.clone(),
            "spike_rate": spike_rate,
            "saturation_ratio": saturation_ratio,
            "R": R_val,
            "N_mean": self.N.float().mean().item(),
            "W_mean": self.W.float().mean().item(),
            "I_mean": self.I.float().mean().item(),
            "mode": effective_mode.value,
            "integration_mode": integration_mode,
            "network_state": self.network_state.value,
            "circadian_encoding_gate": float(self.circadian_encoding_gate.item()),
            "circadian_consolidation_gate": float(self.circadian_consolidation_gate.item()),
            "circadian_attention_gate": float(self.circadian_attention_gate.item()),
            "effective_eta": float(self._latest_policy.effective_eta),
        }

        if dend_contribution is not None:
            out["dend_contribution"] = dend_contribution

        return out

    def step(
        self,
        x: torch.Tensor,
        reward: Optional[float] = None,
        dt: float = 1.0,
        mode: Optional[Union[LearningMode, str]] = None,
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
        self._apply_runtime_injections()

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
        post_rate_float = float(stats.post_rate)

        if x_mean is None:
            self.logger.warning(
                "Sem x_mean acumulado; abortando aplicação de plasticidade"
            )
            return

        cfg = self.cfg
        device = self.theta.device
        x_mean = x_mean.to(device)

        post_rate_float = max(0.0, min(1.0, post_rate_float))
        post_rate_t = torch.tensor([post_rate_float], device=device)

        saturation_ratio = self._compute_saturation_ratio()

        if cfg.neuromod_mode == "external":
            R = float(reward) if reward is not None else 0.0
            R = max(-1.0, min(1.0, R))
        else:
            R = self._compute_R_endogenous(post_rate_float, saturation_ratio)
        R_t = torch.tensor([R], device=device)

        # Atualiza cada dendrito
        try:
            for d_idx, dend in enumerate(self.dendrites):
                active_mask = (x_mean[d_idx] > cfg.activity_threshold).float()
                # `x_mean[d_idx]` já é média temporal por sinapse (E[x_j]).
                # Dividir novamente por `n_active` acopla sinapses entre si e
                # reduz indevidamente o termo local pré-sináptico, quebrando a
                # equivalência ONLINE vs BATCH da regra Hebbiana baseada em taxa.
                pre_rate = x_mean[d_idx] * active_mask
                pre_rate = pre_rate.clamp(0.0, 1.0)

                dend.update_synapses_rate_based(
                    pre_rate=pre_rate,
                    post_rate=post_rate_t,
                    R=R_t,
                    dt=dt,
                    mode=self.mode,
                )
        except Exception as exc:
            self.logger.error(f"Falha ao aplicar plasticidade: {exc}")
            raise

        self.stats_acc.reset()
        self.invalidate_weight_cache()
        self.logger.debug(
            f"✅ Plasticidade aplicada, R={R:.3f}, post_rate={post_rate_float:.3f}"
        )

        if self.telemetry is not None and self.telemetry.enabled():
            with self._telemetry_lock:
                try:
                    sid = self._safe_step_id()
                    self.telemetry.emit(
                        commit_event(
                            step_id=sid,
                            mode=self.mode.value,
                            committed=True,
                            post_rate=post_rate_float,
                            R=R,
                        )
                    )
                except Exception as exc:
                    self.logger.error(f"Falha ao emitir evento de commit: {exc}")

    # ========== CICLO DE SONO ==========

    @torch.no_grad()
    def sleep(self, duration: float = 60.0) -> None:
        """Ciclo de sono: consolida sinapses."""
        self.logger.info(f"💤 Iniciando sono por {duration}ms")
        self.sleep_count.add_(1)

        self.run_sleep_cycle(duration=duration)

        self.logger.info("✅ Sono concluído")

        if self.telemetry is not None and self.telemetry.enabled():
            with self._telemetry_lock:
                try:
                    sid = self._safe_step_id()
                    self.telemetry.emit(
                        sleep_event(
                            step_id=sid,
                            mode=self.mode.value,
                            duration=float(duration),
                        )
                    )
                except Exception as exc:
                    self.logger.error(f"Falha ao emitir evento de sleep: {exc}")

    # ========== UTILITÁRIOS ==========

    def to(self, device: torch.device) -> "MPJRDNeuron":
        """Move neurônio para device e valida consistência."""
        super().to(device)
        self.logger.debug(f"📦 Neurônio movido para {device}")
        return self

    # ========== MÉTRICAS ==========

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas consolidadas do neurônio em uma única passagem.

        Returns
        -------
        Dict[str, Any]
            Dicionário com estatísticas globais de estado do neurônio.
        """

        device = self.theta.device
        q_probs = getattr(self, "_metrics_n_quantile_probs", None)
        if q_probs is None or q_probs.device != device:
            q_probs = torch.tensor((0.25, 0.5, 0.75), device=device)
            self._metrics_n_quantile_probs = q_probs

        n_count = 0
        n_sum = 0.0
        n_sumsq = 0.0
        n_min = float("inf")
        n_max = float("-inf")

        l_count = 0
        l_sum = 0.0

        i_count = 0
        i_sum = 0.0
        i_sumsq = 0.0
        i_min = float("inf")
        i_max = float("-inf")

        w_count = 0
        w_sum = 0.0
        w_positive = 0

        prot_count = 0
        prot_sum = 0.0

        for dend in self.dendrites:
            syn_batch = getattr(dend, "synapse_batch", None)
            if syn_batch is not None:
                n_flat = syn_batch.N.float().reshape(-1)
                l_flat = getattr(syn_batch, "L", syn_batch.N).float().reshape(-1)
                i_flat = syn_batch.I.float().reshape(-1)
                w_flat = syn_batch.W.float().reshape(-1)
                prot_flat = syn_batch.protection.float().reshape(-1)

                n_count += int(n_flat.numel())
                n_sum += float(n_flat.sum().item())
                n_sumsq += float((n_flat * n_flat).sum().item())
                n_min = min(n_min, float(n_flat.min().item()))
                n_max = max(n_max, float(n_flat.max().item()))

                l_count += int(l_flat.numel())
                l_sum += float(l_flat.sum().item())

                i_count += int(i_flat.numel())
                i_sum += float(i_flat.sum().item())
                i_sumsq += float((i_flat * i_flat).sum().item())
                i_min = min(i_min, float(i_flat.min().item()))
                i_max = max(i_max, float(i_flat.max().item()))

                w_count += int(w_flat.numel())
                w_sum += float(w_flat.sum().item())
                w_positive += int((w_flat > 0).sum().item())

                prot_count += int(prot_flat.numel())
                prot_sum += float(prot_flat.sum().item())
                continue

            for syn in dend.synapses:
                n_flat = syn.N.float().reshape(-1)
                l_flat = syn.L.float().reshape(-1)
                i_flat = syn.I.float().reshape(-1)
                w_flat = syn.W.float().reshape(-1)
                prot_flat = syn.protection.float().reshape(-1)

                n_count += int(n_flat.numel())
                n_sum += float(n_flat.sum().item())
                n_sumsq += float((n_flat * n_flat).sum().item())
                n_min = min(n_min, float(n_flat.min().item()))
                n_max = max(n_max, float(n_flat.max().item()))

                l_count += int(l_flat.numel())
                l_sum += float(l_flat.sum().item())

                i_count += int(i_flat.numel())
                i_sum += float(i_flat.sum().item())
                i_sumsq += float((i_flat * i_flat).sum().item())
                i_min = min(i_min, float(i_flat.min().item()))
                i_max = max(i_max, float(i_flat.max().item()))

                w_count += int(w_flat.numel())
                w_sum += float(w_flat.sum().item())
                w_positive += int((w_flat > 0).sum().item())

                prot_count += int(prot_flat.numel())
                prot_sum += float(prot_flat.sum().item())

        if n_count:
            n_mean = n_sum / n_count
            n_var = max((n_sumsq / n_count) - (n_mean * n_mean), 0.0)
            n_std = n_var**0.5

            if self.dendrites:
                first_batch = getattr(self.dendrites[0], "synapse_batch", None)
                if first_batch is not None:
                    all_n = torch.cat([d.synapse_batch.N.float().reshape(-1) for d in self.dendrites])
                else:
                    all_n = torch.stack([syn.N.float().reshape(()) for d in self.dendrites for syn in d.synapses])
                percentiles = torch.quantile(all_n, q_probs)
            else:
                percentiles = torch.zeros(3, device=device)
        else:
            percentiles = torch.zeros(3, device=device)
            n_count = 1
            n_mean = 0.0
            n_std = 0.0
            n_min = 0.0
            n_max = 0.0

        l_mean = (l_sum / l_count) if l_count else 0.0

        if i_count:
            i_mean = i_sum / i_count
            i_var = max((i_sumsq / i_count) - (i_mean * i_mean), 0.0)
            i_std = i_var**0.5
        else:
            i_mean = 0.0
            i_std = 0.0
            i_min = 0.0
            i_max = 0.0

        w_mean = (w_sum / w_count) if w_count else 0.0
        active_synapse_ratio = (w_positive / w_count) if w_count else 0.0
        protection_ratio = (prot_sum / prot_count) if prot_count else 0.0

        metrics = {
            "type": "MPJRDNeuron",
            "id": id(self),
            "theta": self.theta.item(),
            "r_hat": self.r_hat.item(),
            "step_count": self.homeostasis.step_count.item(),
            "N_mean": n_mean,
            "L_mean": l_mean,
            "N_std": n_std,
            "N_min": n_min,
            "N_max": n_max,
            "N_25p": percentiles[0].item(),
            "N_median": percentiles[1].item(),
            "N_75p": percentiles[2].item(),
            "I_mean": i_mean,
            "I_std": i_std,
            "I_min": i_min,
            "I_max": i_max,
            "W_mean": w_mean,
            "saturation_ratio": self._compute_saturation_ratio(),
            "protection_ratio": protection_ratio,
            "total_synapses": int(n_count),
            "total_dendrites": len(self.dendrites),
            "mode": self.mode.value,
            "mode_switches": self.mode_switches.item(),
            "sleep_count": self.sleep_count.item(),
            "has_pending_updates": self.stats_acc.has_data,
            "weight_cache_enabled": self._weight_cache_enabled,
            "weight_cache_dirty": self._weight_cache_dirty,
            "weight_cache_rebuilds": self._weight_cache_rebuilds,
            "pending_count": (
                self.stats_acc.acc_count.item() if self.stats_acc.has_data else 0
            ),
            "device": str(self.theta.device),
            "homeostasis_stable": self.homeostasis.is_stable(),
            "network_state": self.network_state.value,
            "global_time_ms": float(self.global_time_ms.item()),
            "circadian_phase": float(self.circadian_phase.item()),
            "circadian_encoding_gate": float(self.circadian_encoding_gate.item()),
            "circadian_consolidation_gate": float(self.circadian_consolidation_gate.item()),
            "circadian_attention_gate": float(self.circadian_attention_gate.item()),
            "effective_eta": float(self._latest_policy.effective_eta),
            "effective_attention_gain": float(self._latest_policy.effective_attention_gain),
            "effective_competition_gain": float(self._latest_policy.effective_competition_gain),
            "effective_replay_priority": float(self._latest_policy.effective_replay_priority),
            "effective_consolidation_rate": float(self._latest_policy.effective_consolidation_rate),
            "effective_decay_rate": float(self._latest_policy.effective_decay_rate),
            "dopamine_like_signal": float(self._latest_neuromodulatory_state.dopamine_like_signal),
            "acetylcholine_like_signal": float(self._latest_neuromodulatory_state.acetylcholine_like_signal),
            "norepinephrine_like_signal": float(self._latest_neuromodulatory_state.norepinephrine_like_signal),
            "serotonin_like_signal": float(self._latest_neuromodulatory_state.serotonin_like_signal),
            "learning_progress_score": float(self._ema_reward.item() * self._ema_spike_rate.item()),
            "reward_alignment_score": float(1.0 - abs(self._ema_reward.item() - self._ema_spike_rate.item())),
            "plasticity_utilization": float(min(1.0, self._latest_policy.effective_eta / max(self._base_i_eta, 1e-6))),
            "useful_update_ratio": float(self._latest_policy.effective_attention_gain / (1.0 + self._latest_policy.effective_competition_gain)),
            "representational_growth": float(n_std),
            "active_synapse_ratio": float(active_synapse_ratio),
        }

        return metrics

    def get_audit_trace_snapshot(self) -> Dict[str, torch.Tensor]:
        """Return a snapshot of the on-device circular step trace.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with winner dendrite indices and associated signal.
        """
        return {
            "winner_idx": self._trace_winner_idx.clone(),
            "winner_signal": self._trace_signal.clone(),
            "pointer": self._trace_ptr.clone(),
        }


    @torch.no_grad()
    def _phase_pruning_gate(self) -> float:
        if self.cfg.pruning_strategy != "phase_scheduled":
            return 1.0
        phase_rad = torch.deg2rad(self.circadian_phase.to(dtype=torch.float32))
        gate = 0.5 * (1.0 + torch.cos(phase_rad))
        return float((1.0 - self.cfg.pruning_schedule_strength) + self.cfg.pruning_schedule_strength * gate.item())

    @torch.no_grad()
    def _refresh_pruning_mask(self) -> None:
        if not bool(getattr(self.cfg, "pruning_enabled", True)):
            self.pruning_mask.fill_(1.0)
            return

        if self.cfg.pruning_strategy == "static":
            threshold = float(getattr(self.cfg, "pruning_runtime_threshold", 0.05))
        else:
            threshold = float(getattr(self.cfg, "pruning_runtime_threshold", 0.05)) * self._phase_pruning_gate()

        keep = (self._get_consolidated_weights(self.theta.device).abs() >= threshold).to(self.pruning_mask.dtype)
        self.pruning_mask.copy_(keep)

    @torch.no_grad()
    def _consolidate_pruning_from_runtime(self) -> None:
        self._refresh_pruning_mask()

    @torch.no_grad()
    def collect_connectivity_snapshot(self) -> Dict[str, torch.Tensor]:
        effective = self.connectivity_mask * self.pruning_mask
        return {
            "connectivity_mask": self.connectivity_mask.clone(),
            "effective_connectivity": effective.clone(),
            "active_by_dendrite": effective.sum(dim=1).clone(),
            "active_ratio": effective.mean().reshape(1).clone(),
        }

    @torch.no_grad()
    def collect_pruning_snapshot(self) -> Dict[str, torch.Tensor]:
        pruned = (self.pruning_mask <= 0).to(self.pruning_mask.dtype)
        return {
            "pruning_mask": self.pruning_mask.clone(),
            "pruned_by_dendrite": pruned.sum(dim=1).clone(),
            "pruned_ratio": pruned.mean().reshape(1).clone(),
        }

    @torch.no_grad()
    def collect_phase_activity_report(self) -> Dict[str, torch.Tensor]:
        bins = torch.arange(self.phase_activity_hist.numel(), device=self.phase_activity_hist.device)
        baseline = self.phase_activity_hist.mean().expand_as(self.phase_activity_hist)
        active_phase_idx = torch.argmax(self.phase_activity_hist).reshape(1)
        return {
            "phase_bins": bins.clone(),
            "activity": self.phase_activity_hist.clone(),
            "baseline": baseline.clone(),
            "active_phase_idx": active_phase_idx.clone(),
            "delta_vs_baseline": (self.phase_activity_hist - baseline).clone(),
        }

    @torch.no_grad()
    def collect_engram_report(self) -> Dict[str, torch.Tensor]:
        return {
            "resonance_by_dendrite": self._runtime_resonance_cache.clone(),
            "mean_resonance": self._runtime_resonance_cache.mean().reshape(1).clone(),
            "max_resonance": self._runtime_resonance_cache.max().reshape(1).clone(),
        }

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode.value}, D={self.cfg.n_dendrites}, "
            f"S={self.cfg.n_synapses_per_dendrite}, θ={self.theta.item():.2f}, "
            f"device={self.theta.device}"
        )
