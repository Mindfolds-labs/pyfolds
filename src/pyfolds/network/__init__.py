"""Módulo de redes neurais MPJRD"""

from .network import MPJRDNetwork
from .wave_network import MPJRDWaveNetwork

__all__ = [
    "MPJRDNetwork",
    "MPJRDWaveNetwork",
]

__version__ = "2.0.0"  # ✅ Alinhado com core