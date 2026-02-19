import torch
import pyfolds
from pyfolds.core import MPJRDConfig


def _cfg(**kwargs):
    return MPJRDConfig(
        n_dendrites=1,
        n_synapses_per_dendrite=1,
        theta_init=0.01,
        theta_min=0.001,
        theta_max=6.0,
        dendrite_integration_mode="wta_hard",
        adaptation_increment=0.0,
        adaptation_max=0.0,
        t_refrac_abs=2.0,
        t_refrac_rel=5.0,
        **kwargs,
    )


def test_absolute_refractory_is_inviolable_after_initial_spike():
    if not pyfolds.ADVANCED_AVAILABLE:
        return
    neuron = pyfolds.MPJRDNeuronAdvanced(_cfg())

    for syn in neuron.dendrites[0].synapses:
        syn.N.fill_(neuron.cfg.n_max)
    neuron.dendrites[0]._invalidate_cache()
    neuron.u_stp.fill_(1.0)
    neuron.R_stp.fill_(1.0)
    x = torch.ones(1, 1, 1) * 10.0
    out_t0 = neuron.forward(x, dt=1.0, collect_stats=False, mode="online")
    assert out_t0["spikes"].item() == 1.0

    out_t1 = neuron.forward(x, dt=1.0, collect_stats=False, mode="online")
    assert out_t1["refrac_blocked"].item() is True
    assert out_t1["spikes"].item() == 0.0
