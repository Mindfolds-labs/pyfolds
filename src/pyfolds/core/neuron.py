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
        name: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg

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
        self.register_buffer("mode_switches", torch.tensor(0))
        self.register_buffer("sleep_count", torch.tensor(0))
        self.register_buffer("_theta_cap_buf", torch.zeros(1))
        self.logger.debug(f"   ✅ Modo inicial: {self.mode.value}")

        # ===== THREAD SAFETY PARA TELEMETRIA =====
        self._telemetry_lock = Lock()

        # ===== FILA LOCK-FREE DE INJEÇÕES RUNTIME (MindControl) =====
        self._runtime_injections: Queue[tuple[str, Any]] = Queue(
            maxsize=max(1, int(getattr(cfg, "runtime_queue_maxsize", 2048)))
        )

        # ===== TELEMETRIA =====
        self.register_buffer("step_id", torch.tensor(0, dtype=torch.int64))
        self.register_buffer(
            "_theta_cap_buf",
            torch.zeros(1, dtype=self.theta.dtype, device=self.theta.device),
        )
        if TELEMETRY_AVAILABLE and enable_telemetry:
            from ..telemetry import (
                TelemetryConfig,
                TelemetryController,
                TelemetryProfile,
            )

            profile_enum = (
                TelemetryProfile(telemetry_profile)
                if isinstance(telemetry_profile, str)
                else telemetry_profile
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

        # Valida devices após inicialização
        self._validate_internal_devices()

        self._gradient_hook_handles = []
        self._install_gradient_health_monitor()

        self.logger.info(f"✅ Neurônio {name or id(self)} inicializado com sucesso")

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

        expected_device = self.theta.device
        if self.step_id.device != expected_device:
            raise RuntimeError(
                f"step_id device {self.step_id.device} != theta device {expected_device}"
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
                self._refresh_config_references()
                applied += 1

        return applied

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
        post_rate_t = torch.tensor(
            [max(0.0, min(1.0, post_rate))], device=self.theta.device
        )

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
        return torch.einsum("bds,ds->bd", x, weights)

    @validate_input(
        expected_ndim=3,
        expected_shape_fn=lambda self: (
            self.cfg.n_dendrites,
            self.cfg.n_synapses_per_dendrite,
        ),
    )
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

        self._apply_runtime_injections()

        # Valida device
        self._validate_input_device(x)

        device = self.theta.device
        B, D, _ = x.shape
        self._set_refractory_state(False)

        # ===== 1. INTEGRAÇÃO DENDRÍTICA =====
        use_vectorized_dendrites = getattr(self.cfg, "use_vectorized_dendrites", True)
        if use_vectorized_dendrites:
            v_dend = self._compute_dendritic_potentials_vectorized(x)
        else:
            dendrite_outputs = [
                dend(x[:, d_idx, :]) for d_idx, dend in enumerate(self.dendrites)
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

        if getattr(self.cfg, "backprop_enabled", True) and hasattr(
            self, "dendrite_amplification"
        ):
            max_gain = getattr(self.cfg, "backprop_max_gain", 2.0)
            amp = (1.0 + self.dendrite_amplification.to(device)).unsqueeze(0)
            amp = amp.clamp(1.0, max_gain)
            v_dend = v_dend * amp

        # ===== 2-4. INTEGRAÇÃO DENDRÍTICA → POTENCIAL → DISPARO =====
        integration_mode = getattr(self.cfg, "dendrite_integration_mode", "wta_hard")

        theta_eff = self.theta
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
            u = dend_out.u
            gated = dend_out.v_nmda
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

            gated = torch.sigmoid(v_dend - (theta_eff * 0.5))
            u = gated.sum(dim=1)
            dend_contribution = None
        else:
            max_idx = v_dend.max(dim=1, keepdim=True)[1]
            gated = torch.zeros_like(v_dend)
            gated.scatter_(1, max_idx, v_dend.gather(1, max_idx))
            u = gated.sum(dim=1)
            dend_contribution = None

        u_raw = u
        spikes = (u >= theta_eff).float()

        # ===== 5. ESTATÍSTICAS =====
        spike_rate = spikes.mean().item()
        saturation_ratio = (
            self._compute_saturation_ratio()
            if self.cfg.neuromod_mode == "capacity"
            else 0.0
        )

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
            and not self.cfg.defer_updates
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
            self.step_id.add_(1)
            sid = int(self.step_id.item())

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
                    sid = int(self.step_id.item())
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

        for i, dend in enumerate(self.dendrites):
            self.logger.debug(f"   Dendrito {i}: consolidando...")
            dend.consolidate(dt=duration)
        self.invalidate_weight_cache()

        self.logger.info("✅ Sono concluído")

        if self.telemetry is not None and self.telemetry.enabled():
            with self._telemetry_lock:
                try:
                    sid = int(self.step_id.item())
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
        """Retorna métricas consolidadas do neurônio."""
        N_flat = self.N.float().flatten()
        I_flat = self.I.flatten()

        if len(N_flat) > 0:
            percentiles = torch.quantile(N_flat, torch.tensor([0.25, 0.5, 0.75]))
        else:
            percentiles = torch.tensor([0.0, 0.0, 0.0])

        metrics = {
            "type": "MPJRDNeuron",
            "id": id(self),
            "theta": self.theta.item(),
            "r_hat": self.r_hat.item(),
            "step_count": self.homeostasis.step_count.item(),
            "N_mean": N_flat.mean().item(),
            "L_mean": self.L.float().mean().item(),
            "N_std": N_flat.std().item(),
            "N_min": N_flat.min().item(),
            "N_max": N_flat.max().item(),
            "N_25p": percentiles[0].item(),
            "N_median": percentiles[1].item(),
            "N_75p": percentiles[2].item(),
            "I_mean": I_flat.mean().item(),
            "I_std": I_flat.std().item(),
            "I_min": I_flat.min().item(),
            "I_max": I_flat.max().item(),
            "saturation_ratio": self._compute_saturation_ratio(),
            "protection_ratio": self.protection.float().mean().item(),
            "total_synapses": self.N.numel(),
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
        }

        self.logger.debug(
            f"📊 Métricas coletadas: N_mean={metrics['N_mean']:.1f}, θ={metrics['theta']:.2f}"
        )
        return metrics

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode.value}, D={self.cfg.n_dendrites}, "
            f"S={self.cfg.n_synapses_per_dendrite}, θ={self.theta.item():.2f}, "
            f"device={self.theta.device}"
        )
