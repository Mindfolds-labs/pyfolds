"""Tests for MPJRDNeuronV2."""

import torch
from concurrent.futures import ThreadPoolExecutor

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


def test_step_id_thread_safe_increment_v2(small_config):
    neuron = pyfolds.MPJRDNeuronV2(small_config)
    x = torch.randn(1, small_config.n_dendrites, small_config.n_synapses_per_dendrite)

    calls = 40
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(neuron.forward, x, None, None, False, 1.0) for _ in range(calls)]
        for fut in futures:
            fut.result()

    assert int(neuron.step_id.item()) == calls


def test_theta_eff_caps_unreachable_threshold():
    cfg = pyfolds.MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=2, theta_init=4.5)
    neuron = pyfolds.MPJRDNeuronV2(cfg)

    with torch.no_grad():
        for dend in neuron.dendrites:
            for syn in dend.synapses:
                syn.N.fill_(cfg.n_max)
            dend._invalidate_cache()

    x = 10.0 * torch.ones(16, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
    out = neuron(x, collect_stats=False)

    assert out["somatic"].max().item() > (cfg.n_dendrites - 1e-3) * 0.5
    assert out["spikes"].mean().item() > 0.0


def test_vectorization_and_forward_integrity_batch64():
    """Valida vetorização em lote e estabilidade numérica do forward."""
    cfg = pyfolds.MPJRDConfig(n_dendrites=16, n_synapses_per_dendrite=8)
    neuron = pyfolds.MPJRDNeuronV2(cfg)

    batch_size = 64
    x = torch.rand(batch_size, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
    out = neuron(x, collect_stats=False)

    assert out["spikes"].shape[0] == batch_size
    assert out["u"].shape[0] == batch_size
    assert not torch.isnan(out["u"]).any()
    assert not torch.isnan(out["v_dend"]).any()
