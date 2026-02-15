"""Camada especializada para neurônios MPJRD-Wave."""

from typing import Optional
import torch

from .layer import MPJRDLayer
from ..wave import MPJRDWaveConfig, MPJRDWaveNeuron


class MPJRDWaveLayer(MPJRDLayer):
    """Wrapper de conveniência para construir camadas wave."""

    def __init__(
        self,
        n_neurons: int,
        cfg: MPJRDWaveConfig,
        name: str = "",
        enable_telemetry: bool = False,
        telemetry_profile: str = "off",
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            n_neurons=n_neurons,
            cfg=cfg,
            name=name,
            neuron_cls=MPJRDWaveNeuron,
            enable_telemetry=enable_telemetry,
            telemetry_profile=telemetry_profile,
            device=device,
        )
