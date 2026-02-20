"""Edge-case tests for SpikingNetwork input preparation."""

import torch

import pyfolds


def test_prepare_input_with_single_dendrite_avoids_division_by_zero():
    cfg = pyfolds.NeuronConfig(n_dendrites=1, n_synapses_per_dendrite=4, device="cpu")

    net = pyfolds.SpikingNetwork("single_dendrite")
    net.add_layer("in", pyfolds.AdaptiveNeuronLayer(2, cfg, device=torch.device("cpu")))
    net.add_layer("out", pyfolds.AdaptiveNeuronLayer(3, cfg, device=torch.device("cpu")))
    net.connect("in", "out")
    net.build()

    x = torch.randn(5, 2, 1, 4)
    out = net(x)

    assert out["output"].shape == (5, 3)
