"""Configuração imutável do neurônio MPJRD - SUPORTE A 9 MECANISMOS"""

from dataclasses import dataclass, field, replace
import warnings
import math
from typing import Literal, Dict, Optional, Tuple

# Literals para modos configuráveis
NeuromodMode = Literal["external", "capacity", "surprise"]
InhibitionMode = Literal["lateral", "feedback", "both", "none"]
PlasticityMode = Literal["stdp", "hebbian", "both", "none"]
RefracMode = Literal["absolute", "relative", "both"]
STDPInputSource = Literal["raw", "stp"]
LTDRule = Literal["classic", "current"]
WeightQuantization = Literal["logN", "uniformW"]
StatsAccumulatorMode = Literal["dense", "sparse_masked"]
AuditMode = Literal["off", "light", "full"]
ContractEnforcement = Literal["off", "warn", "strict"]


@dataclass(frozen=True)
class TopologyConfig:
    """Subconfiguração de topologia."""

    n_dendrites: int = 4
    n_synapses_per_dendrite: int = 32
    use_vectorized_synapses: bool = False


@dataclass(frozen=True)
class FilamentConfig:
    """Subconfiguração de filamentos/peso."""

    n_min: int = 0
    n_max: int = 31
    w_scale: float = 5.0
    weight_quantization: WeightQuantization = "logN"
    n_levels: int = 32


@dataclass(frozen=True)
class PlasticityConfig:
    """Subconfiguração de plasticidade."""

    i_eta: float = 0.01
    i_gamma: float = 0.99
    beta_w: float = 0.0
    hebbian_ltd_ratio: float = 1.0
    i_ltp_th: float = 5.0
    i_ltd_th: float = -5.0
    ltd_threshold_saturated: float = -10.0
    i_min: float = -20.0
    i_max: float = 50.0
    i_decay_sleep: float = 0.99
    A_plus: float = 1.0
    A_minus: float = 1.0
    neuromod_scale: float = 1.0
    forgetting_tau: float = 1e12
    forgetting_access_lambda: float = 0.5


@dataclass(frozen=True)
class HomeostasisConfig:
    """Subconfiguração homeostática."""

    theta_init: float = 1.5
    theta_min: float = 0.5
    theta_max: float = 6.0
    homeostasis_alpha: float = 0.1
    homeostasis_eta: float = 0.1
    target_spike_rate: float = 0.1
    dead_neuron_threshold: float = 0.01
    dead_neuron_penalty: float = 1.0
    homeostasis_eps: float = 1e-7
    homeostasis_stability_window: int = 200
    dt: float = 1.0


@dataclass(frozen=True)
class CircadianConfig:
    """Subconfiguração circadiana."""

    circadian_enabled: bool = False
    circadian_cycle_hours: float = 12.0
    circadian_phase_bins: int = 24
    circadian_auto_mode: bool = False
    circadian_sleep_duration: float = 60.0
    circadian_plasticity_min: float = 0.1
    circadian_plasticity_max: float = 1.5
    replay_interval_steps: int = 32
    circadian_day_start_hour: float = 0.0
    circadian_am_cortisol: float = 1.0
    circadian_pm_cortisol: float = 0.3
    circadian_am_melatonin: float = 0.1
    circadian_pm_melatonin: float = 0.9


@dataclass(frozen=True)
class AuditConfig:
    """Subconfiguração de auditoria/telemetria operacional."""

    audit_mode: AuditMode = "off"
    audit_trace_capacity: int = 512
    contract_enforcement: ContractEnforcement = "warn"


@dataclass(frozen=True)
class MPJRDConfig:
    """
    Configuração completa do neurônio MPJRD com 9 mecanismos avançados.
    
    ✅ CONSTANTES CONFIGURÁVEIS ADICIONADAS:
        - i_decay_sleep: Decaimento de I durante sono
        - activity_threshold: Threshold para sinapse ativa
        - homeostasis_eps: Epsilon para homeostase
    """
    
    # ===== TOPOLOGIA =====
    n_dendrites: int = 4
    n_synapses_per_dendrite: int = 32
    use_vectorized_synapses: bool = False
    
    # ===== MECANISMO 1: FILAMENTOS (N) =====
    n_min: int = 0
    n_max: int = 31
    w_scale: float = 5.0
    weight_quantization: WeightQuantization = "logN"
    n_levels: int = 32
    
    # ===== MECANISMO 2: PLASTICIDADE (I) =====
    i_eta: float = 0.01
    i_gamma: float = 0.99
    beta_w: float = 0.0
    hebbian_ltd_ratio: float = 1.0
    i_ltp_th: float = 5.0
    i_ltd_th: float = -5.0
    ltd_threshold_saturated: float = -10.0
    i_min: float = -20.0
    i_max: float = 50.0
    
    # ✅ NOVO: Decaimento de I durante sono/consolidação
    i_decay_sleep: float = 0.99
    
    # ===== MECANISMO 3: DINÂMICA ST (u, R) =====
    u0: float = 0.2
    R0: float = 1.0
    U: float = 0.3
    tau_fac: float = 100.0
    tau_rec: float = 800.0
    saturation_recovery_time: float = 60.0
    
    # ===== MECANISMO 4: HOMEOSTASE =====
    theta_init: float = 1.5
    theta_min: float = 0.5
    theta_max: float = 6.0
    homeostasis_alpha: float = 0.1
    homeostasis_eta: float = 0.1
    target_spike_rate: float = 0.1
    dead_neuron_threshold: float = 0.01
    dead_neuron_penalty: float = 1.0
    
    # ✅ NOVO: Threshold para considerar sinapse ativa
    activity_threshold: float = 0.01
    stats_accumulator_mode: StatsAccumulatorMode = "dense"
    sparse_min_activity_ratio: float = 0.15
    scientific_debug_stats: bool = False
    enable_accumulator_profiling: bool = False
    enable_weight_cache: bool = True
    
    # ✅ NOVO: Epsilon para homeostase (evita divisão por zero)
    homeostasis_eps: float = 1e-7
    homeostasis_stability_window: int = 200

    # ===== INTEGRAÇÃO DENDRÍTICA (substitui WTA hard) =====
    dendrite_integration_mode: str = "nmda_shunting"
    dendrite_gain: float = 2.0
    theta_dend_ratio: float = 0.25
    shunting_eps: float = 0.1
    shunting_strength: float = 1.0
    bap_proportional: bool = True
    
    # ===== MECANISMO 5: BACKPROPAGAÇÃO =====
    backprop_enabled: bool = True
    backprop_delay: float = 2.0
    backprop_signal: float = 0.5
    backprop_amp_tau: float = 20.0
    backprop_trace_tau: float = 10.0
    backprop_max_amp: float = 0.4
    backprop_max_gain: float = 2.0
    backprop_active_threshold: float = 0.1
    backprop_queue_maxlen: Optional[int] = None
    
    # ===== MECANISMO 6: ADAPTAÇÃO (SFA) =====
    adaptation_enabled: bool = True
    adaptation_increment: float = 0.8
    adaptation_decay: float = 0.99
    adaptation_max: float = 5.0
    adaptation_tau: float = 50.0
    
    # ===== MECANISMO 7: REFRATÁRIO =====
    refrac_mode: RefracMode = "both"
    t_refrac_abs: float = 2.0
    t_refrac_rel: float = 5.0
    refrac_rel_strength: float = 3.0
    
    # ===== MECANISMO 8: INIBIÇÃO (populacional) =====
    inhibition_mode: InhibitionMode = "both"
    lateral_strength: float = 0.3
    feedback_strength: float = 0.5
    inhibition_sigma: float = 2.0
    n_excitatory: int = 100
    n_inhibitory: int = 25
    target_sparsity: float = 0.03
    
    # ===== MECANISMO 9: STDP =====
    plasticity_mode: PlasticityMode = "both"
    tau_pre: float = 20.0
    tau_post: float = 20.0
    A_plus: float = 0.01
    A_minus: float = 0.012
    stdp_trace_threshold: float = 0.01
    stdp_input_source: STDPInputSource = "raw"
    ltd_rule: LTDRule = "current"
    stdp_consolidation_scale: float = 1.0
    max_eligibility: float = 1e6

    # ===== THRESHOLDS E REPRODUTIBILIDADE =====
    spike_threshold: float = 0.5
    random_seed: Optional[int] = None
    active_synapses_ratio: float = 0.25

    # ===== MECANISMO WAVE (OPCIONAL) =====
    base_frequency: float = 12.0
    frequency_step: float = 4.0
    class_frequencies: Optional[Tuple[float, ...]] = None
    phase_decay: float = 0.98
    phase_buffer_size: int = 32
    phase_sensitivity: float = 1.0
    phase_plasticity_gain: float = 0.25
    dendritic_threshold: float = 0.0
    latency_scale: float = 1.0
    amplitude_eps: float = 1e-6
    circadian_enabled: bool = False
    circadian_cycle_hours: float = 12.0
    circadian_day_start_hour: float = 6.0
    circadian_phase_bins: int = 24
    circadian_am_cortisol: float = 1.0
    circadian_pm_cortisol: float = 0.4
    circadian_am_melatonin: float = 0.2
    circadian_pm_melatonin: float = 0.8
    circadian_auto_mode: bool = False
    circadian_sleep_duration: float = 60.0
    circadian_plasticity_min: float = 0.1
    circadian_plasticity_max: float = 1.5
    replay_interval_steps: int = 32
    experimental_circadian_enabled: bool = True

    # ===== INIBIÇÃO =====
    inhibition_trainable_i2e: bool = False
    
    # ===== NEUROMODULAÇÃO =====
    neuromod_mode: NeuromodMode = "surprise"
    neuromod_scale: float = 1.0
    forgetting_tau: float = 1e12
    forgetting_access_lambda: float = 0.5
    
    # Capacidade
    cap_k_sat: float = 1.2
    cap_k_rate: float = 0.8
    cap_bias: float = 0.0
    
    # Surpresa
    sup_k: float = 2.0
    sup_bias: float = 0.0
    
    # ===== EXECUÇÃO =====
    plastic: bool = True
    defer_updates: bool = True
    consolidation_rate: float = 0.1
    tau_consolidation: float = 1.0
    distributed_sync_on_consolidate: bool = True
    eps: float = 1e-8
    dt: float = 1.0
    device: str = "auto"
    runtime_queue_maxsize: int = 2048

    # ===== CONECTIVIDADE E PODA ESTRUTURAL =====
    pruning_enabled: bool = True
    pruning_strategy: str = "static"  # static | phase_scheduled | replay_consolidated
    consolidate_pruning_after_replay: bool = False
    pruning_schedule_strength: float = 1.0
    pruning_runtime_threshold: float = 0.05

    # ===== ESTABILIDADE NUMÉRICA =====
    max_log_weight: float = 10.0
    float_precision: str = "float32"
    numerical_stability_checks: bool = True

    # ===== WAVE (OSCILAÇÃO COMO MECANISMO) =====
    wave_enabled: bool = False
    experimental_wave_enabled: bool = True
    wave_n_frequencies: int = 8
    wave_base_frequency: float = 10.0
    wave_frequency_step: float = 5.0
    wave_phase_sensitivity: float = 1.0
    wave_phase_decay: float = 0.98
    wave_phase_buffer_size: int = 32
    wave_learning_rate_gain: float = 1.0
    wave_focus_gain: float = 1.0
    wave_excitation_gain: float = 1.0
    wave_stability_gain: float = 1.0
    wave_sleep_consolidation: bool = True
    wave_sleep_replay_rate: float = 0.1
    wave_sleep_pruning_threshold: float = 0.01

    # ===== NEURAL SPEECH TRACKING (OPCIONAL) =====
    enable_speech_envelope_tracking: bool = False
    speech_envelope_method: str = "gammatone"
    enable_phase_reset_on_audio_event: bool = False
    phase_reset_threshold: float = 0.25
    phase_reset_target: float = 0.0
    enable_cross_frequency_coupling: bool = False
    enable_spatial_latency_gradient: bool = False
    spatial_latency_max_ms: float = 100.0
    spatial_latency_scale: float = 1.0

    # ===== CONFIGURAÇÕES DO NOETIC =====
    max_engrams: int = 10_000_000
    pruning_threshold: float = 0.1
    engram_n_frequencies: int = 8
    experimental_engram_enabled: bool = True
    experimental_engram_indexing_enabled: bool = True
    experimental_engram_cache_enabled: bool = True
    enable_specialization: bool = True
    synthesis_threshold: float = 0.6
    sleep_cycle_hours: float = 24.0
    replay_batch_size: int = 32
    model_name: str = "Noetic"
    save_checkpoints: bool = True
    checkpoint_interval: int = 86400

    # ===== AUDITORIA OPERACIONAL =====
    audit_mode: AuditMode = "off"
    audit_trace_capacity: int = 512
    contract_enforcement: ContractEnforcement = "warn"

    # Sub-configs compostos (mantém compatibilidade flat)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    filament: FilamentConfig = field(default_factory=FilamentConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    homeostasis: HomeostasisConfig = field(default_factory=HomeostasisConfig)
    circadian: CircadianConfig = field(default_factory=CircadianConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    _attr_cache: Dict[str, object] = field(default_factory=dict, init=False, repr=False, compare=False)
    _subfield_owner_cache: Dict[str, object] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validações pós-inicialização."""
        try:
            import torch
            object.__setattr__(self, '_torch', torch)
        except ImportError:
            object.__setattr__(self, '_torch', None)
            warnings.warn("PyTorch não encontrado", RuntimeWarning)
        
        object.__setattr__(self, "topology", TopologyConfig(self.n_dendrites, self.n_synapses_per_dendrite))
        object.__setattr__(self, "filament", FilamentConfig(self.n_min, self.n_max, self.w_scale, self.weight_quantization, self.n_levels))
        object.__setattr__(self, "plasticity", PlasticityConfig(
            i_eta=self.i_eta,
            i_gamma=self.i_gamma,
            beta_w=self.beta_w,
            hebbian_ltd_ratio=self.hebbian_ltd_ratio,
            i_ltp_th=self.i_ltp_th,
            i_ltd_th=self.i_ltd_th,
            ltd_threshold_saturated=self.ltd_threshold_saturated,
            i_min=self.i_min,
            i_max=self.i_max,
            i_decay_sleep=self.i_decay_sleep,
            A_plus=self.A_plus,
            A_minus=self.A_minus,
            neuromod_scale=self.neuromod_scale,
            forgetting_tau=self.forgetting_tau,
            forgetting_access_lambda=self.forgetting_access_lambda,
        ))
        object.__setattr__(self, "homeostasis", HomeostasisConfig(
            theta_init=self.theta_init,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            homeostasis_alpha=self.homeostasis_alpha,
            homeostasis_eta=self.homeostasis_eta,
            target_spike_rate=self.target_spike_rate,
            dead_neuron_threshold=self.dead_neuron_threshold,
            dead_neuron_penalty=self.dead_neuron_penalty,
            homeostasis_eps=self.homeostasis_eps,
            homeostasis_stability_window=self.homeostasis_stability_window,
            dt=self.dt,
        ))
        object.__setattr__(self, "circadian", CircadianConfig(
            circadian_enabled=self.circadian_enabled,
            circadian_cycle_hours=self.circadian_cycle_hours,
            circadian_phase_bins=self.circadian_phase_bins,
            circadian_auto_mode=self.circadian_auto_mode,
            circadian_sleep_duration=self.circadian_sleep_duration,
            circadian_day_start_hour=self.circadian_day_start_hour,
            circadian_am_cortisol=self.circadian_am_cortisol,
            circadian_pm_cortisol=self.circadian_pm_cortisol,
            circadian_am_melatonin=self.circadian_am_melatonin,
            circadian_pm_melatonin=self.circadian_pm_melatonin,
            circadian_plasticity_min=self.circadian_plasticity_min,
            circadian_plasticity_max=self.circadian_plasticity_max,
        ))
        object.__setattr__(self, "audit", AuditConfig(
            audit_mode=self.audit_mode,
            audit_trace_capacity=self.audit_trace_capacity,
            contract_enforcement=self.contract_enforcement,
        ))

        # Resolve device
        if self.device == "auto":
            if self._torch is not None and self._torch.cuda.is_available():
                object.__setattr__(self, 'device', 'cuda')
            else:
                object.__setattr__(self, 'device', 'cpu')
        elif self.device == "cuda":
            if self._torch is None or not self._torch.cuda.is_available():
                warnings.warn("CUDA não disponível, usando CPU", RuntimeWarning)
                object.__setattr__(self, 'device', 'cpu')
        
        # Validações básicas
        if self.n_min >= self.n_max:
            raise ValueError(f"n_min ({self.n_min}) must be < n_max ({self.n_max})")
        
        if self.theta_min >= self.theta_max:
            raise ValueError(f"theta_min ({self.theta_min}) must be < theta_max ({self.theta_max})")
        
        if self.i_min >= self.i_max:
            raise ValueError(f"i_min ({self.i_min}) must be < i_max ({self.i_max})")
        
        if self.tau_fac <= 0:
            raise ValueError(f"tau_fac must be positive, got {self.tau_fac}")
        
        if self.tau_rec <= 0:
            raise ValueError(f"tau_rec must be positive, got {self.tau_rec}")
        
        if self.activity_threshold <= 0:
            raise ValueError(f"activity_threshold must be > 0, got {self.activity_threshold}")

        if self.stats_accumulator_mode not in {"dense", "sparse_masked"}:
            raise ValueError(
                "stats_accumulator_mode inválido: "
                f"{self.stats_accumulator_mode}. Use: 'dense' ou 'sparse_masked'"
            )

        if not 0.0 <= self.sparse_min_activity_ratio <= 1.0:
            raise ValueError(
                "sparse_min_activity_ratio deve estar em [0, 1], "
                f"recebido {self.sparse_min_activity_ratio}"
            )
        
        if self.homeostasis_eps <= 0:
            raise ValueError(f"homeostasis_eps must be > 0, got {self.homeostasis_eps}")

        if self.neuromod_scale <= 0:
            raise ValueError(f"neuromod_scale must be > 0, got {self.neuromod_scale}")

        if self.forgetting_tau <= 0:
            raise ValueError(
                f"forgetting_tau deve ser > 0, recebido {self.forgetting_tau}"
            )

        if self.forgetting_access_lambda < 0:
            raise ValueError(
                "forgetting_access_lambda deve ser >= 0, "
                f"recebido {self.forgetting_access_lambda}"
            )

        if self.hebbian_ltd_ratio < 0:
            raise ValueError(
                "hebbian_ltd_ratio deve ser >= 0, "
                f"recebido {self.hebbian_ltd_ratio}"
            )

        valid_integration_modes = {"wta_hard", "wta_soft", "nmda_shunting"}
        if self.dendrite_integration_mode not in valid_integration_modes:
            raise ValueError(
                "dendrite_integration_mode inválido: "
                f"{self.dendrite_integration_mode}. "
                f"Use: {valid_integration_modes}"
            )

        if self.dendrite_gain <= 0:
            raise ValueError(f"dendrite_gain deve ser > 0, recebido {self.dendrite_gain}")

        if not 0.0 < self.theta_dend_ratio < 1.0:
            raise ValueError(
                "theta_dend_ratio deve estar em (0, 1), "
                f"recebido {self.theta_dend_ratio}"
            )

        if self.shunting_eps <= 0:
            raise ValueError(f"shunting_eps deve ser > 0, recebido {self.shunting_eps}")

        if self.shunting_strength < 0:
            raise ValueError(
                "shunting_strength deve ser >= 0, "
                f"recebido {self.shunting_strength}"
            )

        if not (0.0 < self.active_synapses_ratio <= 1.0):
            raise ValueError(
                f"active_synapses_ratio must be in (0, 1], got {self.active_synapses_ratio}"
            )

        if self.circadian_cycle_hours <= 0:
            raise ValueError(
                "circadian_cycle_hours must be > 0, "
                f"got {self.circadian_cycle_hours}"
            )

        if self.circadian_phase_bins <= 0:
            raise ValueError(
                f"circadian_phase_bins must be > 0, got {self.circadian_phase_bins}"
            )

        if self.circadian_sleep_duration <= 0:
            raise ValueError(
                "circadian_sleep_duration must be > 0, "
                f"got {self.circadian_sleep_duration}"
            )

        if self.stdp_input_source not in {"raw", "stp"}:
            raise ValueError(
                "stdp_input_source inválido: "
                f"{self.stdp_input_source}. Use: 'raw' ou 'stp'"
            )

        if self.ltd_rule not in {"classic", "current"}:
            raise ValueError(
                "ltd_rule inválido: "
                f"{self.ltd_rule}. Use: 'classic' ou 'current'"
            )

        if self.phase_buffer_size <= 0:
            raise ValueError("phase_buffer_size must be > 0")
        if self.base_frequency <= 0:
            raise ValueError("base_frequency must be > 0")

        if self.speech_envelope_method not in {"hilbert", "gammatone"}:
            raise ValueError("speech_envelope_method must be 'hilbert' or 'gammatone'")
        if self.phase_reset_threshold < 0:
            raise ValueError("phase_reset_threshold must be >= 0")
        if self.spatial_latency_max_ms < 0:
            raise ValueError("spatial_latency_max_ms must be >= 0")
        if self.spatial_latency_scale <= 0:
            raise ValueError("spatial_latency_scale must be > 0")
        if self.frequency_step < 0:
            raise ValueError("frequency_step must be >= 0")
        if self.phase_decay <= 0 or self.phase_decay > 1:
            raise ValueError("phase_decay must be in (0, 1]")
        if self.amplitude_eps <= 0:
            raise ValueError("amplitude_eps must be > 0")

        if self.class_frequencies is not None:
            if len(self.class_frequencies) == 0:
                raise ValueError("class_frequencies cannot be empty")
            if any(f <= 0 for f in self.class_frequencies):
                raise ValueError("all class_frequencies must be > 0")

        if self.weight_quantization not in {"logN", "uniformW"}:
            raise ValueError(
                "weight_quantization inválido: "
                f"{self.weight_quantization}. Use: 'logN' ou 'uniformW'"
            )

        if self.wave_n_frequencies <= 0:
            raise ValueError("wave_n_frequencies must be > 0")

        if self.wave_base_frequency <= 0:
            raise ValueError("wave_base_frequency must be > 0")

        if self.wave_frequency_step < 0:
            raise ValueError("wave_frequency_step must be >= 0")

        if not 0 < self.wave_phase_decay <= 1:
            raise ValueError("wave_phase_decay must be in (0, 1]")

        if self.wave_phase_buffer_size <= 0:
            raise ValueError("wave_phase_buffer_size must be > 0")

        if self.backprop_queue_maxlen is not None and self.backprop_queue_maxlen <= 0:
            raise ValueError("backprop_queue_maxlen must be > 0")

        if self.wave_sleep_replay_rate < 0:
            raise ValueError("wave_sleep_replay_rate must be >= 0")

        if self.wave_sleep_pruning_threshold < 0:
            raise ValueError("wave_sleep_pruning_threshold must be >= 0")

        valid_pruning_strategies = {"static", "phase_scheduled", "replay_consolidated"}
        if self.pruning_strategy not in valid_pruning_strategies:
            raise ValueError(
                "pruning_strategy inválido: "
                f"{self.pruning_strategy}. Use: {valid_pruning_strategies}"
            )

        if self.pruning_schedule_strength < 0:
            raise ValueError("pruning_schedule_strength must be >= 0")

        if not 0.0 <= self.pruning_runtime_threshold <= 1.0:
            raise ValueError("pruning_runtime_threshold must be in [0, 1]")

        bool_fields = (
            "experimental_wave_enabled",
            "experimental_circadian_enabled",
            "experimental_engram_enabled",
            "experimental_engram_indexing_enabled",
            "experimental_engram_cache_enabled",
        )
        for field_name in bool_fields:
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be bool, got {type(value).__name__}")
        
        # Warnings
        if self.ltd_threshold_saturated > self.i_ltd_th:
            warnings.warn(
                f"ltd_threshold_saturated ({self.ltd_threshold_saturated}) > i_ltd_th ({self.i_ltd_th})",
                RuntimeWarning
            )
        
        if self.homeostasis_eta > 0.1:
            warnings.warn(
                f"homeostasis_eta={self.homeostasis_eta} pode causar instabilidade",
                RuntimeWarning
            )

        self.validate_numerical_safety()

    def validate_numerical_safety(self) -> None:
        """Valida parâmetros críticos para segurança numérica."""
        if self.w_scale <= 0:
            raise ValueError(f"w_scale deve ser > 0, recebido {self.w_scale}")

        if self.n_levels < 2:
            raise ValueError(f"n_levels deve ser >= 2, recebido {self.n_levels}")

        if self.n_max > 2**30:
            raise ValueError(f"n_max={self.n_max} pode causar overflow numérico em log2")

        if self.max_log_weight <= 0:
            raise ValueError(
                f"max_log_weight deve ser > 0, recebido {self.max_log_weight}"
            )

        if self.activity_threshold >= 1e6:
            raise ValueError(
                "activity_threshold muito alto; pode inviabilizar estatísticas por atividade"
            )

        if self.float_precision not in {"float32", "float64"}:
            raise ValueError(
                "float_precision deve ser 'float32' ou 'float64', "
                f"recebido {self.float_precision}"
            )

        if self.audit_mode not in {"off", "light", "full"}:
            raise ValueError(
                "audit_mode deve ser 'off', 'light' ou 'full', "
                f"recebido {self.audit_mode}"
            )

        if self.contract_enforcement not in {"off", "warn", "strict"}:
            raise ValueError(
                "contract_enforcement deve ser 'off', 'warn' ou 'strict', "
                f"recebido {self.contract_enforcement}"
            )

        if self.audit_trace_capacity < 1:
            raise ValueError(
                "audit_trace_capacity deve ser >= 1, "
                f"recebido {self.audit_trace_capacity}"
            )

        max_w = self.w_max
        if max_w > 100.0:
            warnings.warn(
                f"W pode atingir {max_w:.1f}; risco de gradiente instável.",
                RuntimeWarning,
            )

    @property
    def w_max(self) -> float:
        """Peso máximo derivado da lei logarítmica atual."""
        return math.log2(1.0 + self.n_max) / self.w_scale
    
    def get_ts(self, param_name: str) -> float:
        """Retorna constante de tempo em ms para um parâmetro."""
        tau_map = {
            'facilitation': self.tau_fac,
            'recovery': self.tau_rec,
            'backprop_amp': self.backprop_amp_tau,
            'backprop_trace': self.backprop_trace_tau,
            'adaptation': self.adaptation_tau,
            'stdp_pre': self.tau_pre,
            'stdp_post': self.tau_post,
        }
        return tau_map.get(param_name, 0.0)
    
    def get_decay_rate(self, tau: float, dt: Optional[float] = None) -> float:
        """Calcula taxa de decaimento exponencial."""
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")

        if dt is None:
            dt = self.dt

        if dt < 0:
            raise ValueError(f"dt must be >= 0, got {dt}")

        return math.exp(-dt / tau)
    
    def to_dict(self) -> Dict:
        """Converte configuração para dicionário."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MPJRDConfig':
        """Cria configuração a partir de dicionário."""
        return cls(**data)
    
    def get_preset(self, name: str = "default") -> 'MPJRDConfig':
        """Retorna configuração pré-definida."""
        presets = {
            'default': self,
            'fast': MPJRDConfig(
                n_dendrites=2,
                n_synapses_per_dendrite=16,
                tau_fac=50.0,
                tau_rec=400.0,
                homeostasis_eta=0.1,
                dt=2.0
            ),
            'precise': MPJRDConfig(
                n_dendrites=8,
                n_synapses_per_dendrite=64,
                tau_fac=100.0,
                tau_rec=800.0,
                homeostasis_eta=0.02,
                dt=0.5
            ),
            'sparse': MPJRDConfig(
                target_spike_rate=0.05,
                target_sparsity=0.01,
                lateral_strength=0.5,
                feedback_strength=0.7
            ),
            'gpu': MPJRDConfig(
                n_dendrites=16,
                n_synapses_per_dendrite=128,
                device='cuda'
            )
        }
        
        if name not in presets:
            warnings.warn(f"Preset '{name}' não encontrado. Usando default.")
            return presets['default']
        
        return presets[name]
    


    def with_runtime_update(self, **updates: float) -> "MPJRDConfig":
        """Retorna nova config validada com campos atualizados em runtime."""
        return replace(self, **updates)

    def resolve_runtime_alias(self, name: str) -> str:
        """Normaliza aliases de parâmetros usados por controladores externos."""
        aliases = {
            "learning_rate": "i_eta",
            "theta": "theta_init",
        }
        return aliases.get(name, name)


    def validate_combination(self) -> list[Warning]:
        """Retorna avisos para combinações potencialmente problemáticas."""
        out: list[Warning] = []

        if self.wave_enabled and self.wave_n_frequencies < 4:
            out.append(Warning("wave_enabled=True com wave_n_frequencies < 4 pode reduzir resolução espectral"))

        if self.adaptation_enabled and self.adaptation_tau < self.t_refrac_rel:
            out.append(Warning("adaptation_tau < t_refrac_rel pode conflitar entre SFA e refratário"))

        if (
            self.plasticity_mode == "stdp"
            and self.tau_pre == self.tau_post
            and self.A_plus == self.A_minus
        ):
            out.append(Warning("Configuração STDP neutra: tau_pre==tau_post e A_plus==A_minus"))

        if self.circadian_enabled and self.circadian_cycle_hours > 24:
            out.append(Warning("circadian_cycle_hours > 24 é biologicamente incomum"))

        return out

    def __getattr__(self, name: str):
        """Fallback de compatibilidade para acesso flat via subconfigs."""
        if name in self._attr_cache:
            return self._attr_cache[name]

        sub_owner = self._subfield_owner_cache.get(name)
        if sub_owner is None:
            subconfigs = (
                self.topology,
                self.filament,
                self.plasticity,
                self.homeostasis,
                self.circadian,
                self.audit,
            )
            for sub in subconfigs:
                if name in sub.__dataclass_fields__:
                    sub_owner = sub
                    self._subfield_owner_cache[name] = sub
                    break

        if sub_owner is not None:
            value = getattr(sub_owner, name)
            self._attr_cache[name] = value
            return value

        raise AttributeError(
            f"MPJRDConfig has no attribute '{name}'. "
            "Atributo desconhecido para config principal e subconfigs."
        )

    def __repr__(self) -> str:
        """Representação string da configuração."""
        lines = ["MPJRDConfig:"]
        lines.append(f"  Topologia: D={self.n_dendrites}, S={self.n_synapses_per_dendrite}")
        lines.append(f"  Filamentos: [{self.n_min}, {self.n_max}], w_scale={self.w_scale}")
        lines.append(f"  Plasticidade: η={self.i_eta}, γ={self.i_gamma}, β_w={self.beta_w}")
        lines.append(f"  Homeostase: θ∈[{self.theta_min},{self.theta_max}], target={self.target_spike_rate}")
        lines.append(f"  Constantes: activity_th={self.activity_threshold}, eps_homeo={self.homeostasis_eps}")
        return "\n".join(lines)


# Alias operacional v2 sem quebrar identidade científica/serialização legada.
NeuronConfig = MPJRDConfig
