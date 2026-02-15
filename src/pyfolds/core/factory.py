"""Factory central para criação de neurônios."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, Optional, Type
import warnings

from .config import MPJRDConfig


class NeuronType(Enum):
    """Tipos de neurônio suportados pelo factory."""

    STANDARD = "standard"
    WAVE = "wave"
    V2 = "v2"


class NeuronFactory:
    """Factory registry-based para criação desacoplada de neurônios."""

    _registry: Dict[NeuronType, Type] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, neuron_type: NeuronType, neuron_class: Type) -> None:
        """
        Registra uma classe de neurônio no factory.
        
        Args:
            neuron_type: Tipo do neurônio (STANDARD, WAVE, V2)
            neuron_class: Classe a ser registrada
        """
        cls._registry[neuron_type] = neuron_class
        cls._initialized = True

    @classmethod
    def create(cls, neuron_type: NeuronType, cfg: MPJRDConfig, **kwargs: Any):
        """
        Cria uma instância de neurônio do tipo especificado.
        
        Args:
            neuron_type: Tipo do neurônio a ser criado
            cfg: Configuração do neurônio
            **kwargs: Argumentos adicionais para o construtor
            
        Returns:
            Instância do neurônio
            
        Raises:
            ValueError: Se o tipo não estiver registrado
            RuntimeError: Se os imports falharem
        """
        # Garante que os tipos padrão estão registrados
        if not cls._initialized:
            register_default_neurons()
        
        if neuron_type not in cls._registry:
            raise ValueError(
                f"Tipo de neurônio não registrado: {neuron_type.value}. "
                f"Registrados: {[t.value for t in cls._registry.keys()]}"
            )
        
        try:
            return cls._registry[neuron_type](cfg, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Falha ao criar neurônio do tipo {neuron_type.value}: {e}"
            ) from e

    @classmethod
    def is_registered(cls, neuron_type: NeuronType) -> bool:
        """Verifica se um tipo está registrado."""
        return neuron_type in cls._registry


def register_neuron(neuron_type: NeuronType) -> Callable[[Type], Type]:
    """Decorator para registrar classes de neurônio automaticamente."""

    def decorator(neuron_cls: Type) -> Type:
        NeuronFactory.register(neuron_type, neuron_cls)
        return neuron_cls

    return decorator


def register_default_neurons() -> None:
    """Registra tipos padrão de forma lazy para evitar import circular."""
    if NeuronFactory._initialized:
        return

    try:
        from .neuron import MPJRDNeuron
        from .neuron_v2 import MPJRDNeuronV2
        
        # Import wave apenas se disponível
        try:
            from ..wave.neuron import MPJRDWaveNeuron
            NeuronFactory.register(NeuronType.WAVE, MPJRDWaveNeuron)
        except ImportError:
            warnings.warn(
                "Módulo wave não disponível. Tipo WAVE não registrado.",
                ImportWarning
            )
        
        NeuronFactory.register(NeuronType.STANDARD, MPJRDNeuron)
        NeuronFactory.register(NeuronType.V2, MPJRDNeuronV2)
        
    except ImportError as e:
        raise RuntimeError(
            f"Falha ao importar classes de neurônio padrão: {e}"
        ) from e


def infer_neuron_type(cfg: MPJRDConfig) -> NeuronType:
    """Infere tipo de neurônio com base na configuração."""
    try:
        from ..wave.config import MPJRDWaveConfig

        if isinstance(cfg, MPJRDWaveConfig):
            return NeuronType.WAVE
    except ImportError:
        pass
    
    return NeuronType.STANDARD


def get_available_types() -> list[str]:
    """Retorna lista de tipos de neurônio disponíveis."""
    if not NeuronFactory._initialized:
        register_default_neurons()
    return [t.value for t in NeuronFactory._registry.keys()]