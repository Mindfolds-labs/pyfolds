import torch
import pyfolds


def test_bap_amplification_changes_dendritic_computation_and_clamps_gain():
    cfg = pyfolds.MPJRDConfig(backprop_max_gain=1.2)
    base_neuron = pyfolds.MPJRDNeuronAdvanced(cfg)
    amp_neuron = pyfolds.MPJRDNeuronAdvanced(cfg)
    amp_neuron.load_state_dict(base_neuron.state_dict(), strict=False)

    x = torch.ones(1, cfg.n_dendrites, cfg.n_synapses_per_dendrite)

    base_neuron.dendrite_amplification.zero_()
    baseline = base_neuron.forward(x, mode="inference")

    amp_neuron.dendrite_amplification.fill_(5.0)
    amplified = amp_neuron.forward(x, mode="inference")

    ratio = (amplified["v_dend"] / baseline["v_dend"]).mean().item()
    assert ratio > 1.0
    assert ratio <= cfg.backprop_max_gain + 1e-4
