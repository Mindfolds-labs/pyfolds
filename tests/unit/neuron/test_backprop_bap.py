import torch
import pyfolds
from pyfolds.core import MPJRDConfig


def test_bap_amplification_changes_dendritic_potential_and_is_clamped():
    if not pyfolds.ADVANCED_AVAILABLE:
        return
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=1,
        backprop_max_gain=1.2,
    )
    x = torch.ones(1, 2, 1)
    neuron_base = pyfolds.MPJRDNeuronAdvanced(cfg)
    neuron_amp = pyfolds.MPJRDNeuronAdvanced(cfg)

    for neuron in (neuron_base, neuron_amp):
        for d in neuron.dendrites:
            for syn in d.synapses:
                syn.N.fill_(neuron.cfg.n_max)
            d._invalidate_cache()
        neuron.u_stp.fill_(1.0)
        neuron.R_stp.fill_(1.0)

    neuron_base.dendrite_amplification.zero_()
    out_base = neuron_base.forward(x, mode="inference", collect_stats=False)

    neuron_amp.dendrite_amplification = torch.tensor([10.0, 0.0])
    out_amp = neuron_amp.forward(x, mode="inference", collect_stats=False)

    assert out_amp["v_dend"][0, 0] > out_base["v_dend"][0, 0]
    # ganho efetivo maximo = backprop_max_gain
    gain = out_amp["v_dend"][0, 0] / out_base["v_dend"][0, 0]
    assert gain <= torch.tensor(cfg.backprop_max_gain + 1e-6)
