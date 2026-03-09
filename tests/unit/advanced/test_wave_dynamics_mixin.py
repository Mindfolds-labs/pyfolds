import torch

from pyfolds import MPJRDConfig
from pyfolds.advanced import MPJRDNeuronAdvanced


def test_advanced_neuron_wave_outputs_when_enabled():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
    )
    neuron = MPJRDNeuronAdvanced(cfg)
    x = torch.randn(3, 2, 4)
    out = neuron(x, collect_stats=False)

    assert "wave_real" in out
    assert "wave_imag" in out
    assert out["wave_real"].shape == (3,)


def test_advanced_neuron_has_no_wave_outputs_when_disabled():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=False,
    )
    neuron = MPJRDNeuronAdvanced(cfg)
    x = torch.randn(3, 2, 4)
    out = neuron(x, collect_stats=False)

    assert "wave_real" not in out
    assert "phase" not in out
