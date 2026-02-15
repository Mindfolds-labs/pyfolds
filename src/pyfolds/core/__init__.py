"""Core do PyFolds - componentes fundamentais do neurônio MPJRD

Este módulo exporta todos os componentes necessários para construir
e treinar neurônios MPJRD com 9 mecanismos avançados:  # ✅ Atualizado

- Filamentos (N) com LTP/LTD estrutural
- Potencial interno (I) com plasticidade Hebbiana
- Dinâmica de curto prazo (u, R)
- Homeostase adaptativa (theta, r_hat)
- Neuromodulação (3 modos)
- Backpropagação dendrítica
- Adaptação (SFA)
- STDP com traços
- Consolidação two-factor (sono)
- Refratário
- Inibição populacional

Uso básico:
    from pyfolds.core import MPJRDConfig, MPJRDNeuron, create_accumulator
    
    cfg = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=32)
    neuron = MPJRDNeuron(cfg)
    acc = create_accumulator(cfg)
"""

from .config import MPJRDConfig, NeuromodMode
from .neuron import MPJRDNeuron
from .neuron_v2 import MPJRDNeuronV2
from .dendrite import MPJRDDendrite
from .synapse import MPJRDSynapse
from .homeostasis import HomeostasisController
from .neuromodulation import Neuromodulator
from .accumulator import StatisticsAccumulator, create_accumulator_from_config  # ✅ Adicionado

__version__ = "2.0.0"  # ✅ Atualizado para versão 2.0.0

__all__ = [
    # Classes principais
    "MPJRDConfig",
    "MPJRDNeuron",
    "MPJRDNeuronV2",
    "MPJRDDendrite",
    "MPJRDSynapse",
    "HomeostasisController",
    "Neuromodulator",
    "StatisticsAccumulator",
    
    # Tipos úteis
    "NeuromodMode",
    
    # Factory functions
    "create_neuron",
    "create_neuron_v2",
    "create_accumulator",
    "create_accumulator_from_config",  # ✅ Adicionado
    
    # Metadados
    "__version__",
]


# ===== FACTORY FUNCTIONS (CONVENIÊNCIA) =====

def create_neuron(cfg=None, **kwargs):
    """
    Cria neurônio MPJRD com configuração padrão ou personalizada.
    
    Args:
        cfg: MPJRDConfig (se None, cria com kwargs)
        **kwargs: Parâmetros para MPJRDConfig
    
    Returns:
        MPJRDNeuron configurado
    
    Exemplos:
        # Com config explícita
        cfg = MPJRDConfig(n_dendrites=8)
        neuron = create_neuron(cfg)
        
        # Com kwargs
        neuron = create_neuron(n_dendrites=8, target_spike_rate=0.15)
    """
    if cfg is None:
        cfg = MPJRDConfig(**kwargs)
    return MPJRDNeuron(cfg)


def create_neuron_v2(cfg=None, **kwargs):
    """Cria neurônio MPJRD V2 (integração cooperativa)."""
    if cfg is None:
        cfg = MPJRDConfig(**kwargs)
    return MPJRDNeuronV2(cfg)


def create_accumulator(cfg, track_extra: bool = False):
    """
    Cria acumulador de estatísticas a partir da configuração.
    
    Args:
        cfg: MPJRDConfig
        track_extra: Se True, acumula estatísticas extras
    
    Returns:
        StatisticsAccumulator configurado
    
    Exemplo:
        cfg = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=32)
        acc = create_accumulator(cfg, track_extra=True)
    """
    return StatisticsAccumulator(
        n_dendrites=cfg.n_dendrites,
        n_synapses=cfg.n_synapses_per_dendrite,
        eps=cfg.eps,
        track_extra=track_extra
    )


def demo():
    """Função de demonstração rápida."""
    print(f"🧠 PyFolds Core v{__version__}")
    print("=" * 40)
    print("Componentes disponíveis:")
    for name in __all__:
        print(f"  • {name}")
    print("\n📦 Use create_neuron() para criar um neurônio.")
    print("📊 Use create_accumulator() para monitoramento.")


if __name__ == "__main__":
    demo()