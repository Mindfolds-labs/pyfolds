"""Experimental LEIBREG (Reasoning + Engram + Geometry) subsystem."""

from .config import LeibregConfig
from .imagination import Imagination
from .leibniz_layer import LeibnizLayer
from .leibreg_bridge import NoeticLeibregBridge
from .reg_core import ProximityAttention, REGBlock, REGCore, ResonanceAttention
from .sigreg import SIGReg
from .wordspace import WordSpace

__all__ = [
    "LeibregConfig",
    "WordSpace",
    "LeibnizLayer",
    "ResonanceAttention",
    "ProximityAttention",
    "REGBlock",
    "REGCore",
    "Imagination",
    "SIGReg",
    "NoeticLeibregBridge",
]
