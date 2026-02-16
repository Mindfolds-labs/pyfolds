import pytest
import torch

import pyfolds


def test_neuron_forward_validation_ndim(small_config):
    neuron = pyfolds.MPJRDNeuron(small_config)
    x = torch.rand(2, small_config.n_dendrites)

    with pytest.raises(ValueError, match="3 dimensões"):
        neuron(x)


def test_neuron_forward_validation_dtype(small_config):
    neuron = pyfolds.MPJRDNeuron(small_config)
    x = torch.randint(0, 2, (2, small_config.n_dendrites, small_config.n_synapses_per_dendrite))

    with pytest.raises(TypeError, match="ponto flutuante"):
        neuron(x)
