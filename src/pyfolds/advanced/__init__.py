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
from typing import Optional
import logging

from .refractory import RefractoryMixin
from .stdp import STDPMixin
from .adaptation import AdaptationMixin
from .short_term import ShortTermDynamicsMixin
from .backprop import BackpropMixin
from .inhibition import InhibitionLayer, InhibitionMixin

from ..core.neuron import MPJRDNeuron as MPJRDNeuronBase
from ..wave import MPJRDWaveNeuron as MPJRDWaveNeuronBase
from ..layers.layer import MPJRDLayer
from ..utils.logging import get_logger

logger = logging.getLogger(__name__)

__all__ = [
    "RefractoryMixin",
    "STDPMixin",
    "AdaptationMixin",
    "ShortTermDynamicsMixin",
    "BackpropMixin",
    "InhibitionLayer",
    "InhibitionMixin",
    "MPJRDNeuronAdvanced",
    "MPJRDLayerAdvanced",
    "MPJRDWaveNeuronAdvanced",
    "MPJRDWaveLayerAdvanced",
]


class MPJRDNeuronAdvanced(
    # ✅ ORDEM CORRETA:
    BackpropMixin,  # 1º: amplifica entrada (antes de tudo)
    ShortTermDynamicsMixin,  # 2º: modula entrada (temporário)
    STDPMixin,  # 3º: atualiza traços (não afeta forward)
    AdaptationMixin,  # 4º: modifica u (antes de refratário)
    RefractoryMixin,  # 5º: bloqueia spikes (deve ser ÚLTIMO)
    MPJRDNeuronBase,
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
        self._init_advanced_mixins(cfg, is_wave=False)

    def _init_advanced_mixins(self, cfg, is_wave: bool = False) -> None:
        """Inicializa mixins avançados com validações e logging de diagnóstico."""
        logger = getattr(self, "logger", get_logger(f"pyfolds.advanced.{id(self)}"))
        model_kind = "wave" if is_wave else "standard"

        required_attrs = {
            "t_refrac_abs", "t_refrac_rel", "refrac_rel_strength",
            "tau_pre", "tau_post", "A_plus", "A_minus", "plasticity_mode",
            "u0", "R0", "U", "tau_fac", "tau_rec",
        }
        missing = sorted(attr for attr in required_attrs if not hasattr(cfg, attr))
        if missing:
            raise RuntimeError(
                f"Configuração incompleta para mixins avançados ({model_kind}): "
                f"faltando atributos: {', '.join(missing)}"
            )

        logger.debug("🔧 Inicializando mixins avançados (%s)", model_kind)
        try:
            logger.debug("   • RefractoryMixin")
            RefractoryMixin._init_refractory(
                self,
                t_refrac_abs=cfg.t_refrac_abs,
                t_refrac_rel=cfg.t_refrac_rel,
                refrac_rel_strength=cfg.refrac_rel_strength,
            )

            logger.debug("   • STDPMixin")
            STDPMixin._init_stdp(
                self,
                tau_pre=cfg.tau_pre,
                tau_post=cfg.tau_post,
                A_plus=cfg.A_plus,
                A_minus=cfg.A_minus,
                plasticity_mode=cfg.plasticity_mode,
            )

            logger.debug("   • AdaptationMixin")
            AdaptationMixin._init_adaptation(self, cfg)

            logger.debug("   • ShortTermDynamicsMixin")
            ShortTermDynamicsMixin._init_short_term(
                self,
                u0=cfg.u0,
                R0=cfg.R0,
                U=cfg.U,
                tau_fac=cfg.tau_fac,
                tau_rec=cfg.tau_rec,
            )

            logger.debug("   • BackpropMixin")
            BackpropMixin._init_backprop(self, cfg)
            logger.info("✅ Mixins avançados inicializados (%s)", model_kind)
        except AttributeError as exc:
            logger.exception("❌ Falha na inicialização dos mixins avançados (%s)", model_kind)
            raise RuntimeError(f"Falha na inicialização dos mixins avançados: {exc}") from exc
    
    def get_all_advanced_metrics(self) -> dict:
        """
        Retorna métricas de TODOS os mecanismos avançados.

        Returns:
            dict: Dicionário com métricas de todos os mecanismos
        """
        metrics = super().get_metrics()

        if hasattr(self, "get_refractory_metrics"):
            metrics.update(self.get_refractory_metrics())
        if hasattr(self, "get_adaptation_metrics"):
            metrics.update(self.get_adaptation_metrics())
        if hasattr(self, "get_short_term_metrics"):
            metrics.update(self.get_short_term_metrics())
        if hasattr(self, "get_backprop_metrics"):
            metrics.update(self.get_backprop_metrics())

        # Métricas STDP
        metrics["trace_pre_mean"] = (
            getattr(self, "trace_pre", torch.tensor(0.0)).mean().item()
        )
        metrics["trace_post_mean"] = (
            getattr(self, "trace_post", torch.tensor(0.0)).mean().item()
        )

        return metrics

    def reset_all_mechanisms(self):
        """Reseta todos os mecanismos avançados."""
        if hasattr(self, "reset_refractory"):
            self.reset_refractory()
        if hasattr(self, "reset_stdp_traces"):
            self.reset_stdp_traces()
        if hasattr(self, "reset_adaptation"):
            self.reset_adaptation()
        if hasattr(self, "reset_short_term_dynamics"):
            self.reset_short_term_dynamics()
        if hasattr(self, "reset_backprop"):
            self.reset_backprop()


class MPJRDWaveNeuronAdvanced(
    BackpropMixin,
    ShortTermDynamicsMixin,
    STDPMixin,
    AdaptationMixin,
    RefractoryMixin,
    MPJRDWaveNeuronBase,
):
    """Versão avançada do neurônio wave com mixins de mecanismos v2.x."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        MPJRDNeuronAdvanced._init_advanced_mixins(self, cfg, is_wave=True)


class MPJRDLayerAdvanced(MPJRDLayer):
    """Camada que injeta automaticamente `MPJRDNeuronAdvanced`."""

    def __init__(
        self,
        n_neurons: int,
        cfg,
        name: str = "",
        enable_telemetry: bool = False,
        telemetry_profile: str = "off",
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            n_neurons=n_neurons,
            cfg=cfg,
            name=name,
            neuron_cls=MPJRDNeuronAdvanced,
            enable_telemetry=enable_telemetry,
            telemetry_profile=telemetry_profile,
            device=device,
        )


class MPJRDWaveLayerAdvanced(MPJRDLayer):
    """Camada avançada para neurônios wave + mecanismos v2.x."""

    def __init__(
        self,
        n_neurons: int,
        cfg,
        name: str = "",
        enable_telemetry: bool = False,
        telemetry_profile: str = "off",
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            n_neurons=n_neurons,
            cfg=cfg,
            name=name,
            neuron_cls=MPJRDWaveNeuronAdvanced,
            enable_telemetry=enable_telemetry,
            telemetry_profile=telemetry_profile,
            device=device,
        )
