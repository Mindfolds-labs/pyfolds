import torch
import pyfolds
from pyfolds.core import MPJRDConfig


def test_sfa_is_applied_before_threshold_for_spikes():
    if not pyfolds.ADVANCED_AVAILABLE:
        return
    cfg = MPJRDConfig(
        n_dendrites=1,
        n_synapses_per_dendrite=1,
        theta_init=1.0,
        theta_min=0.5,
        theta_max=6.0,
        adaptation_increment=0.0,
    )
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)
    neuron.adaptation_current = torch.tensor([10.0])

    x = torch.ones(1, 1, 1) * 10.0
    out = neuron.forward(x, dt=1.0)

    assert "u_raw" in out and "u_adapted" in out
    assert out["u_raw"].item() > out["u_adapted"].item()
    assert out["spikes"].item() == 0.0
