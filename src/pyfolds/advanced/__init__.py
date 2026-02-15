# pyfolds/advanced/__init__.py
"""
Módulo de Mecanismos Avançados para PyFolds

Fornece mixins para:
- Refratário (refractory)
- STDP (stdp)
- Adaptação (adaptation)
- Short-term dynamics (short_term)
- Backpropagation (backprop)
- Inibição (inhibition)
"""

import torch  # ✅ Necessário para as métricas

from .refractory import RefractoryMixin
from .stdp import STDPMixin
from .adaptation import AdaptationMixin
from .short_term import ShortTermDynamicsMixin
from .backprop import BackpropMixin
from .inhibition import InhibitionLayer, InhibitionMixin

from ..core.neuron import MPJRDNeuron as MPJRDNeuronBase

__all__ = [
    "RefractoryMixin",
    "STDPMixin",
    "AdaptationMixin",
    "ShortTermDynamicsMixin",
    "BackpropMixin",
    "InhibitionLayer",
    "InhibitionMixin",
    "MPJRDNeuronAdvanced",
]


class MPJRDNeuronAdvanced(
    # ✅ ORDEM CORRETA:
    BackpropMixin,          # 1º: amplifica entrada (antes de tudo)
    ShortTermDynamicsMixin, # 2º: modula entrada (temporário)
    STDPMixin,              # 3º: atualiza traços (não afeta forward)
    AdaptationMixin,        # 4º: modifica u (antes de refratário)
    RefractoryMixin,        # 5º: bloqueia spikes (deve ser ÚLTIMO)
    MPJRDNeuronBase
):
    """
    Neurônio MPJRD com TODOS os mecanismos avançados.
    
    ✅ ORDEM CORRETA DE EXECUÇÃO:
        1. Backpropagation: amplifica entrada baseado em spikes anteriores
        2. Short-term dynamics: modula entrada (facilitação/depressão)
        3. STDP: atualiza traços (não afeta forward atual)
        4. Adaptation: modifica u (corrente de adaptação)
        5. Refractory: bloqueia spikes (deve ser o último)
        6. Base: forward base (WTA, homeostase, etc)
    
    Para inibição, use MPJRDLayerAdvanced em conjunto.
    """
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        # ===== INICIALIZAÇÃO DOS MIXINS =====
        # ✅ CORRIGIDO: passa cfg para os mixins que dependem dele
        
        # Refractory (usa parâmetros individuais)
        if hasattr(self, '_init_refractory'):
            self._init_refractory(
                t_refrac_abs=cfg.t_refrac_abs,
                t_refrac_rel=cfg.t_refrac_rel,
                refrac_rel_strength=cfg.refrac_rel_strength
            )
        
        # STDP (usa parâmetros individuais)
        if hasattr(self, '_init_stdp'):
            self._init_stdp(
                tau_pre=cfg.tau_pre,
                tau_post=cfg.tau_post,
                A_plus=cfg.A_plus,
                A_minus=cfg.A_minus,
                plasticity_mode=cfg.plasticity_mode
            )
        
        # Adaptation (cfg-based)
        if hasattr(self, '_init_adaptation'):
            self._init_adaptation(cfg)  # ✅ passa cfg completo
        
        # Short-term (usa parâmetros individuais)
        if hasattr(self, '_init_short_term'):
            self._init_short_term(
                u0=cfg.u0,
                R0=cfg.R0,
                U=cfg.U,
                tau_fac=cfg.tau_fac,
                tau_rec=cfg.tau_rec
            )
        
        # Backprop (cfg-based)
        if hasattr(self, '_init_backprop'):
            self._init_backprop(cfg)  # ✅ passa cfg completo
    
    def get_all_advanced_metrics(self) -> dict:
        """
        Retorna métricas de TODOS os mecanismos avançados.
        
        Returns:
            dict: Dicionário com métricas de todos os mecanismos
        """
        metrics = super().get_metrics()
        
        if hasattr(self, 'get_refractory_metrics'):
            metrics.update(self.get_refractory_metrics())
        if hasattr(self, 'get_adaptation_metrics'):
            metrics.update(self.get_adaptation_metrics())
        if hasattr(self, 'get_short_term_metrics'):
            metrics.update(self.get_short_term_metrics())
        if hasattr(self, 'get_backprop_metrics'):
            metrics.update(self.get_backprop_metrics())
        
        # Métricas STDP
        metrics['trace_pre_mean'] = getattr(self, 'trace_pre', torch.tensor(0.0)).mean().item()
        metrics['trace_post_mean'] = getattr(self, 'trace_post', torch.tensor(0.0)).mean().item()
        
        return metrics
    
    def reset_all_mechanisms(self):
        """Reseta todos os mecanismos avançados."""
        if hasattr(self, 'reset_refractory'):
            self.reset_refractory()
        if hasattr(self, 'reset_stdp_traces'):
            self.reset_stdp_traces()
        if hasattr(self, 'reset_adaptation'):
            self.reset_adaptation()
        if hasattr(self, 'reset_short_term_dynamics'):
            self.reset_short_term_dynamics()
        if hasattr(self, 'reset_backprop'):
            self.reset_backprop()