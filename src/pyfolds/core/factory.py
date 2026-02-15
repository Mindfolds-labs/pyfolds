"""Factory central para criação de neurônios."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, Optional, Type

from .config import MPJRDConfig


class NeuronType(Enum):
    """Tipos de neurônio suportados pelo factory."""

    STANDARD = "standard"
    WAVE = "wave"
    V2 = "v2"


class NeuronFactory:
    """Factory registry-based para criação desacoplada de neurônios."""

    _registry: Dict[NeuronType, Type] = {}

    @classmethod
    def register(cls, neuron_type: NeuronType, neuron_class: Type) -> None:
        cls._registry[neuron_type] = neuron_class

    @classmethod
    def create(cls, neuron_type: NeuronType, cfg: MPJRDConfig, **kwargs: Any):
        if neuron_type not in cls._registry:
            raise ValueError(f"Tipo de neurônio não registrado: {neuron_type.value}")
        return cls._registry[neuron_type](cfg, **kwargs)


def register_neuron(neuron_type: NeuronType) -> Callable[[Type], Type]:
    """Decorator para registrar classes de neurônio automaticamente."""

    def decorator(neuron_cls: Type) -> Type:
        NeuronFactory.register(neuron_type, neuron_cls)
        return neuron_cls

    return decorator


def register_default_neurons() -> None:
    """Registra tipos padrão de forma lazy para evitar import circular."""
    if NeuronFactory._registry:
        return

    from .neuron import MPJRDNeuron
    from .neuron_v2 import MPJRDNeuronV2
    from ..wave.neuron import MPJRDWaveNeuron

    NeuronFactory.register(NeuronType.STANDARD, MPJRDNeuron)
    NeuronFactory.register(NeuronType.WAVE, MPJRDWaveNeuron)
    NeuronFactory.register(NeuronType.V2, MPJRDNeuronV2)


def infer_neuron_type(cfg: MPJRDConfig) -> NeuronType:
    """Infere tipo de neurônio com base na configuração."""
    from ..wave.config import MPJRDWaveConfig

    if isinstance(cfg, MPJRDWaveConfig):
        return NeuronType.WAVE
    return NeuronType.STANDARD

