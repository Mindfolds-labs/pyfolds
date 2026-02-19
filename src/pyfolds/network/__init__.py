"""Módulo de redes neurais MPJRD"""

from .network import MPJRDNetwork
from .wave_network import MPJRDWaveNetwork
from .builder import NetworkBuilder

__all__ = [
    "MPJRDNetwork",
    "MPJRDWaveNetwork",
    "NetworkBuilder",
]

__version__ = "2.0.0"  # ✅ Alinhado com versão canônica
