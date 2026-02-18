"""Configuração imutável do neurônio MPJRD - SUPORTE A 9 MECANISMOS"""

from dataclasses import dataclass
import warnings
import math
from typing import Literal, Dict, Optional

# Literals para modos configuráveis
NeuromodMode = Literal["external", "capacity", "surprise"]
InhibitionMode = Literal["lateral", "feedback", "both", "none"]
PlasticityMode = Literal["stdp", "hebbian", "both", "none"]
RefracMode = Literal["absolute", "relative", "both"]


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
    
    # ===== MECANISMO 1: FILAMENTOS (N) =====
    n_min: int = 0
    n_max: int = 31
    w_scale: float = 5.0
    
    # ===== MECANISMO 2: PLASTICIDADE (I) =====
    i_eta: float = 0.01
    i_gamma: float = 0.99
    beta_w: float = 0.0
    i_ltp_th: float = 5.0
    i_ltd_th: float = -5.0
    ltd_threshold_saturated: float = -10.0
    i_min: float = -20.0
    i_max: float = 50.0
    
    # ✅ NOVO: Decaimento de I durante sono/consolidação
    i_decay_sleep: float = 0.99
    
    # ===== MECANISMO 3: DINÂMICA ST (u, R) =====
    u0: float = 0.1
    R0: float = 1.0
    U: float = 0.2
    tau_fac: float = 100.0
    tau_rec: float = 800.0
    saturation_recovery_time: float = 60.0
    
    # ===== MECANISMO 4: HOMEOSTASE =====
    theta_init: float = 4.5
    theta_min: float = 2.0
    theta_max: float = 8.0
    homeostasis_alpha: float = 0.1
    homeostasis_eta: float = 0.05
    target_spike_rate: float = 0.1
    dead_neuron_threshold: float = 0.01
    dead_neuron_penalty: float = 0.3
    
    # ✅ NOVO: Threshold para considerar sinapse ativa
    activity_threshold: float = 0.01
    
    # ✅ NOVO: Epsilon para homeostase (evita divisão por zero)
    homeostasis_eps: float = 1e-7

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

    # ===== THRESHOLDS E REPRODUTIBILIDADE =====
    spike_threshold: float = 0.5
    random_seed: Optional[int] = None
    active_synapses_ratio: float = 0.25

    # ===== INIBIÇÃO =====
    inhibition_trainable_i2e: bool = False
    
    # ===== NEUROMODULAÇÃO =====
    neuromod_mode: NeuromodMode = "external"
    neuromod_scale: float = 1.0
    
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
    eps: float = 1e-8
    dt: float = 1.0
    device: str = "auto"

    # ===== ESTABILIDADE NUMÉRICA =====
    max_log_weight: float = 10.0
    float_precision: str = "float32"
    numerical_stability_checks: bool = True
    
    def __post_init__(self):
        """Validações pós-inicialização."""
        try:
            import torch
            object.__setattr__(self, '_torch', torch)
        except ImportError:
            object.__setattr__(self, '_torch', None)
            warnings.warn("PyTorch não encontrado", RuntimeWarning)
        
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
        
        if self.homeostasis_eps <= 0:
            raise ValueError(f"homeostasis_eps must be > 0, got {self.homeostasis_eps}")

        if self.neuromod_scale <= 0:
            raise ValueError(f"neuromod_scale must be > 0, got {self.neuromod_scale}")

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

        if self.n_max > 2**30:
            raise ValueError(f"n_max={self.n_max} pode causar overflow numérico em log2")

        if self.max_log_weight <= 0:
            raise ValueError(
                f"max_log_weight deve ser > 0, recebido {self.max_log_weight}"
            )

        if self.float_precision not in {"float32", "float64"}:
            raise ValueError(
                "float_precision deve ser 'float32' ou 'float64', "
                f"recebido {self.float_precision}"
            )

        max_w = math.log2(1.0 + self.n_max) / self.w_scale
        if max_w > 100.0:
            warnings.warn(
                f"W pode atingir {max_w:.1f}; risco de gradiente instável.",
                RuntimeWarning,
            )
    
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
    
    def __repr__(self) -> str:
        """Representação string da configuração."""
        lines = ["MPJRDConfig:"]
        lines.append(f"  Topologia: D={self.n_dendrites}, S={self.n_synapses_per_dendrite}")
        lines.append(f"  Filamentos: [{self.n_min}, {self.n_max}], w_scale={self.w_scale}")
        lines.append(f"  Plasticidade: η={self.i_eta}, γ={self.i_gamma}, β_w={self.beta_w}")
        lines.append(f"  Homeostase: θ∈[{self.theta_min},{self.theta_max}], target={self.target_spike_rate}")
        lines.append(f"  Constantes: activity_th={self.activity_threshold}, eps_homeo={self.homeostasis_eps}")
        return "\n".join(lines)
