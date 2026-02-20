"""Camadas de neurônios MPJRD para construção de redes neurais"""

from .layer import MPJRDLayer
from .wave_layer import MPJRDWaveLayer

__all__ = [
    "MPJRDLayer",
    "MPJRDWaveLayer",
]

__version__ = "2.0.2"  # ✅ Alinhado com versão canônica
