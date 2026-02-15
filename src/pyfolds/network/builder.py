"""Builder fluente para criação de redes MPJRD."""

from __future__ import annotations

from typing import Optional

from ..core.config import MPJRDConfig
from ..layers.layer import MPJRDLayer
from .network import MPJRDNetwork


class NetworkBuilder:
    """Builder pattern para redes feedforward MPJRD."""

    def __init__(self, name: str = "network"):
        self.network = MPJRDNetwork(name)
        self.last_layer: Optional[str] = None

    def add_layer(
        self,
        name: str,
        n_neurons: int,
        cfg: Optional[MPJRDConfig] = None,
        connect_from_previous: bool = True,
    ) -> "NetworkBuilder":
        """Adiciona camada e conecta automaticamente à camada anterior."""
        if cfg is None:
            cfg = MPJRDConfig()

        layer = MPJRDLayer(n_neurons=n_neurons, cfg=cfg, name=name)
        self.network.add_layer(name, layer)

        if connect_from_previous and self.last_layer is not None:
            self.network.connect(self.last_layer, name)

        self.last_layer = name
        return self

    def build(self) -> MPJRDNetwork:
        """Finaliza a construção validando a topologia."""
        self.network.build()
        return self.network
