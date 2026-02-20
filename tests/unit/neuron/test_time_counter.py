import torch
import pyfolds


def test_time_counter_increments_at_end_of_forward_step():
    cfg = pyfolds.NeuronConfig(t_refrac_abs=2.0)
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)

    x = torch.ones(1, cfg.n_dendrites, cfg.n_synapses_per_dendrite)

    out0 = neuron.forward(x, dt=1.0)
    assert neuron.time_counter.item() == 1.0

    out1 = neuron.forward(x, dt=1.0)
    assert out1["refrac_blocked"][0].item() is True
    assert neuron.time_counter.item() == 2.0

    assert out0["spikes"][0].item() == 1.0
    assert out1["spikes"][0].item() == 0.0
