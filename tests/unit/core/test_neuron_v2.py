"""Tests for MPJRDNeuronV2."""

import torch

import pyfolds


def test_forward_shapes_v2(small_config, batch_size):
    neuron = pyfolds.MPJRDNeuronV2(small_config)
    x = torch.randn(batch_size, small_config.n_dendrites, small_config.n_synapses_per_dendrite)

    out = neuron(x)

    assert out["spikes"].shape == (batch_size,)
    assert out["u"].shape == (batch_size,)
    assert out["somatic"].shape == (batch_size,)
    assert out["v_dend"].shape == (batch_size, small_config.n_dendrites)
    assert out["dendritic_gain"].shape == (batch_size, small_config.n_dendrites)


def test_cooperative_sum_uses_multiple_dendrites():
    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=2, theta_init=1.1)
    neuron = pyfolds.MPJRDNeuronV2(cfg)

    with torch.no_grad():
        for dend in neuron.dendrites:
            for syn in dend.synapses:
                syn.N.fill_(cfg.n_max)
            dend._invalidate_cache()

    x = torch.ones(1, 2, 2)
    out = neuron(x, collect_stats=False)

    # Com entrada forte nos dois dendritos, o ganho cooperativo soma contribuições.
    assert torch.all(out["dendritic_gain"] > 0.5)
    assert out["somatic"].item() > 1.1
    assert out["spikes"].item() == 1.0
