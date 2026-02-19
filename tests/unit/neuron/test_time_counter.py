import torch
import pyfolds
from pyfolds.core import MPJRDConfig


def test_time_counter_updates_at_end_of_forward_step():
    if not pyfolds.ADVANCED_AVAILABLE:
        return
    cfg = MPJRDConfig(
        n_dendrites=1,
        n_synapses_per_dendrite=1,
        t_refrac_abs=2.0,
        theta_init=0.01,
        theta_min=0.001,
        dendrite_integration_mode="wta_hard",
    )
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)

    assert neuron.time_counter.item() == 0.0
    for syn in neuron.dendrites[0].synapses:
        syn.N.fill_(neuron.cfg.n_max)
    neuron.dendrites[0]._invalidate_cache()
    neuron.u_stp.fill_(1.0)
    neuron.R_stp.fill_(1.0)
    x = torch.ones(1, 1, 1) * 10.0

    out = neuron.forward(x, dt=1.0, collect_stats=False, mode="online")
    assert out["spikes"].item() == 1.0
    assert neuron.last_spike_time.item() == 0.0
    assert neuron.time_counter.item() == 1.0
