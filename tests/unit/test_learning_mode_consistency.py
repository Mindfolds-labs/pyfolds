import torch
import pytest

import pyfolds
from pyfolds.core import MPJRDConfig
from pyfolds.layers import MPJRDLayer
from pyfolds.network import MPJRDNetwork
from pyfolds.utils.types import LearningMode


def _make_layer() -> MPJRDLayer:
    cfg = MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=3)
    return MPJRDLayer(n_neurons=2, cfg=cfg)


def test_learning_mode_single_source() -> None:
    assert pyfolds.LearningMode is LearningMode


def test_learning_mode_from_string() -> None:
    assert LearningMode("online") == LearningMode.ONLINE


def test_layer_accepts_mode_string() -> None:
    layer = _make_layer()
    x = torch.rand(4, 2, 2, 3)
    out = layer(x, mode="online")
    assert out["spikes"].shape == (4, 2)


def test_layer_accepts_mode_enum() -> None:
    layer = _make_layer()
    x = torch.rand(4, 2, 2, 3)
    out = layer(x, mode=LearningMode.ONLINE)
    assert out["spikes"].shape == (4, 2)


def test_layer_rejects_invalid_mode() -> None:
    layer = _make_layer()
    x = torch.rand(2, 2, 2, 3)
    with pytest.raises(ValueError, match="mode inválido"):
        layer(x, mode="invalido")


def test_network_accepts_mode_string() -> None:
    cfg = MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=3)
    layer = MPJRDLayer(n_neurons=2, cfg=cfg)
    net = MPJRDNetwork().add_layer("input", layer).build()
    x = torch.rand(4, 2, 2, 3)
    out = net(x, mode="online")
    assert out["output"].shape == (4, 2)


def test_network_rejects_invalid_mode() -> None:
    cfg = MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=3)
    layer = MPJRDLayer(n_neurons=2, cfg=cfg)
    net = MPJRDNetwork().add_layer("input", layer).build()
    x = torch.rand(4, 2, 2, 3)
    with pytest.raises(ValueError, match="mode inválido"):
        net(x, mode="xx")
