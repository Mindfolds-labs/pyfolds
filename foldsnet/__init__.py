"""Pacote principal da arquitetura FOLDSNet."""

from .factory import create_foldsnet
from .model import FOLDSNet

__all__ = ["FOLDSNet", "create_foldsnet"]
