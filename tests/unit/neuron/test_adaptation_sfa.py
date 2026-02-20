import torch
import pyfolds


def test_sfa_applies_before_threshold_and_reduces_spiking_probability():
    cfg = pyfolds.NeuronConfig()
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)
    x = torch.ones(1, cfg.n_dendrites, cfg.n_synapses_per_dendrite)

    neuron._ensure_adaptation_current(1, torch.device("cpu"))
    neuron.adaptation_current.fill_(10.0)

    out = neuron.forward(x, dt=1.0)

    assert out["u_raw"][0].item() > out["u"][0].item()
    assert out["u_eff"][0].item() == out["u"][0].item()
    assert out["spikes"][0].item() == 0.0
