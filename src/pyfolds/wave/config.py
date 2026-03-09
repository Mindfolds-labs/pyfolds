"""Configurações para o modelo MPJRD-Wave (v3.0)."""

from dataclasses import dataclass

from ..core.config import MPJRDConfig


@dataclass(frozen=True)
class MPJRDWaveConfig(MPJRDConfig):
    """Extensão de configuração para habilitar dinâmica WAVE por padrão."""

    wave_enabled: bool = True
