"""Configurações para o modelo MPJRD-Wave (v3.0)."""

from dataclasses import dataclass
from typing import Optional, Tuple

from ..core.config import MPJRDConfig


@dataclass(frozen=True)
class MPJRDWaveConfig(MPJRDConfig):
    """Extensão de configuração para codificação por fase/frequência.

    Este neurônio implementa codificação temporal via oscilações (WAVE)
    usando fase e frequência para representação distribuída de informação.

    Referências:
        Brette, R. (2012). Computing with neurons.
        Körding, K. P., & König, P. (2000).
    """

    wave_enabled: bool = True
    base_frequency: float = 12.0
    frequency_step: float = 4.0
    class_frequencies: Optional[Tuple[float, ...]] = None

    phase_decay: float = 0.98
    phase_buffer_size: int = 32
    phase_sensitivity: float = 1.0
    phase_plasticity_gain: float = 0.25

    dendritic_threshold: float = 0.0
    latency_scale: float = 1.0
    amplitude_eps: float = 1e-6

    def __post_init__(self):
        super().__post_init__()

        if self.phase_buffer_size <= 0:
            raise ValueError("phase_buffer_size must be > 0")
        if self.base_frequency <= 0:
            raise ValueError("base_frequency must be > 0")
        if self.frequency_step < 0:
            raise ValueError("frequency_step must be >= 0")
        if self.phase_decay <= 0 or self.phase_decay > 1:
            raise ValueError("phase_decay must be in (0, 1]")
        if self.amplitude_eps <= 0:
            raise ValueError("amplitude_eps must be > 0")

        if self.class_frequencies is not None:
            if len(self.class_frequencies) == 0:
                raise ValueError("class_frequencies cannot be empty")
            if any(f <= 0 for f in self.class_frequencies):
                raise ValueError("all class_frequencies must be > 0")
