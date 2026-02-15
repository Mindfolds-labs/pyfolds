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
    assert layer.neuron_cls is MPJRDNeuronV2


def test_layer_rejects_invalid_neuron_cls():
    cfg = MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)

    with pytest.raises(TypeError):
        MPJRDLayer(n_neurons=2, cfg=cfg, neuron_cls=torch.nn.Linear)
