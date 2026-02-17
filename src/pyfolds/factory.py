"""Factory para criação de neurônios com API extensível."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Type, Union

from .core.base import BaseNeuron
from .core.config import MPJRDConfig
from .core.neuron import MPJRDNeuron
from .wave.neuron import MPJRDWaveNeuron


class NeuronType(str, Enum):
    """Tipos nativos de neurônios suportados pela fábrica."""

    BASIC = "basic"
    ADVANCED = "advanced"
    WAVE = "wave"


class NeuronFactory:
    """Factory pattern para criação de neurônios PyFolds."""

    _registry: Dict[str, Type[BaseNeuron]] = {}

    @classmethod
    def register(cls, name: str, neuron_class: Type[BaseNeuron]) -> None:
        """Registra um tipo customizado de neurônio."""
        cls._registry[name] = neuron_class

    @classmethod
    def create(
        cls,
        neuron_type: Union[NeuronType, str],
        cfg: MPJRDConfig,
        **kwargs,
    ) -> BaseNeuron:
        """Cria uma instância de neurônio por tipo."""
        type_name = neuron_type.value if isinstance(neuron_type, NeuronType) else str(neuron_type)

        if type_name == NeuronType.BASIC.value:
            return MPJRDNeuron(cfg, **kwargs)

        if type_name == NeuronType.ADVANCED.value:
            from .advanced import MPJRDNeuronAdvanced

            return MPJRDNeuronAdvanced(cfg, **kwargs)

        if type_name == NeuronType.WAVE.value:
            return MPJRDWaveNeuron(cfg, **kwargs)

        if type_name in cls._registry:
            neuron_cls = cls._registry[type_name]
            try:
                return neuron_cls(cfg, **kwargs)
            except TypeError as e:
                raise ValueError(
                    f"Erro ao criar neurônio {type_name}: {e}\n"
                    f"kwargs inválidos: {list(kwargs.keys())}"
                ) from e

        raise ValueError(f"Tipo de neurônio desconhecido: {type_name}")
