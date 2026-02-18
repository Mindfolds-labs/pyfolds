"""Módulo de redes neurais MPJRD"""

from .network import MPJRDNetwork
from .wave_network import MPJRDWaveNetwork
from .builder import NetworkBuilder

__all__ = [
    "MPJRDNetwork",
    "MPJRDWaveNetwork",
    "NetworkBuilder",
]

__version__ = "1.0.1"  # ✅ Alinhado com core
