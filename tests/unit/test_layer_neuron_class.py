"""Tests for MPJRDLayer with custom neuron class."""

import pytest
import torch

from pyfolds import MPJRDConfig
from pyfolds.core import MPJRDNeuronV2
from pyfolds.layers import MPJRDLayer


def test_layer_accepts_neuron_v2():
    cfg = MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    layer = MPJRDLayer(n_neurons=3, cfg=cfg, neuron_cls=MPJRDNeuronV2)

    x = torch.randn(5, 3, 2, 4)
    out = layer(x)

    assert out["spikes"].shape == (5, 3)
    assert out["u"].shape == (5, 3)
    assert out["u_values"].shape == (5, 3)
    assert layer.neuron_cls is MPJRDNeuronV2


def test_layer_rejects_invalid_neuron_cls():
    cfg = MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)

    with pytest.raises(TypeError):
        MPJRDLayer(n_neurons=2, cfg=cfg, neuron_cls=torch.nn.Linear)


def test_layer_has_no_legacy_neuron_class_attr():
    cfg = MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    layer = MPJRDLayer(n_neurons=3, cfg=cfg)
    assert not hasattr(layer, "neuron_class")


def test_layer_forwards_dt_to_neurons(monkeypatch):
    cfg = MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=2)
    layer = MPJRDLayer(n_neurons=2, cfg=cfg)

    seen_dt = []

    def _wrap_forward(orig):
        def _inner(x, reward=None, mode=None, dt=1.0, **kwargs):
            seen_dt.append(dt)
            return orig(x, reward=reward, mode=mode, dt=dt, **kwargs)

        return _inner

    for neuron in layer.neurons:
        monkeypatch.setattr(neuron, 'forward', _wrap_forward(neuron.forward))

    x = torch.randn(3, 2, 2, 2)
    layer(x, dt=0.25)

    assert seen_dt == [0.25, 0.25]
