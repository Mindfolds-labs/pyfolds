"""Tests for input/device validation contract on neuron and layer."""

import pytest
import torch

from pyfolds import NeuronConfig, AdaptiveNeuronLayer
from pyfolds.core.neuron import MPJRDNeuron


def _cfg() -> NeuronConfig:
    return NeuronConfig(n_dendrites=2, n_synapses_per_dendrite=4)


def test_neuron_rejects_non_tensor_input() -> None:
    neuron = MPJRDNeuron(_cfg())
    with pytest.raises(TypeError, match="torch.Tensor"):
        neuron._validate_input_device("invalid")


def test_neuron_rejects_input_from_different_device() -> None:
    neuron = MPJRDNeuron(_cfg())
    x_meta = torch.empty((1, 2, 4), device="meta")
    with pytest.raises(RuntimeError, match="Input device"):
        neuron._validate_input_device(x_meta)


def test_layer_prepare_input_accepts_supported_shapes() -> None:
    cfg = _cfg()
    layer = AdaptiveNeuronLayer(n_neurons=3, cfg=cfg)

    x_full = torch.rand(5, 3, 2, 4)
    assert layer._prepare_input(x_full).shape == (5, 3, 2, 4)

    x_bnd = torch.rand(5, 3, 2)
    assert layer._prepare_input(x_bnd).shape == (5, 3, 2, 4)

    x_bds = torch.rand(5, 2, 4)
    assert layer._prepare_input(x_bds).shape == (5, 3, 2, 4)


def test_layer_prepare_input_rejects_invalid_shape() -> None:
    cfg = _cfg()
    layer = AdaptiveNeuronLayer(n_neurons=3, cfg=cfg)
    with pytest.raises(ValueError, match="Formato de entrada não suportado"):
        layer._prepare_input(torch.rand(5, 3))
