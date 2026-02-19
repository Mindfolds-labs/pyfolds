import torch
import pyfolds


def test_absolute_refractory_is_inviolable():
    cfg = pyfolds.MPJRDConfig(t_refrac_abs=3.0, t_refrac_rel=5.0)
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)

    x = torch.ones(1, cfg.n_dendrites, cfg.n_synapses_per_dendrite)

    first = neuron.forward(x, dt=1.0)
    assert first["spikes"][0].item() == 1.0

    neuron.time_counter.fill_(1.0)
    neuron.last_spike_time = torch.tensor([0.0])

    second = neuron.forward(x, dt=1.0)

    assert second["refrac_blocked"][0].item() is True
    assert second["spikes"][0].item() == 0.0
