"""Rede utilitária para MPJRD-Wave."""

from ..layers import MPJRDWaveLayer
from ..wave import MPJRDWaveConfig
from .network import MPJRDNetwork


class MPJRDWaveNetwork(MPJRDNetwork):
    """Wrapper para construção de redes com camadas wave."""

    def add_wave_layer(self, name: str, n_neurons: int, cfg: MPJRDWaveConfig) -> "MPJRDWaveNetwork":
        self.add_layer(name, MPJRDWaveLayer(n_neurons=n_neurons, cfg=cfg, name=name))
        return self
