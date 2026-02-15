"""Camadas de neurônios MPJRD para construção de redes neurais"""

from .layer import MPJRDLayer
from .wave_layer import MPJRDWaveLayer

__all__ = [
    "MPJRDLayer",
    "MPJRDWaveLayer",
]

__version__ = "2.0.0"  # ✅ Alinhado com core