import torch
import pyfolds
from pyfolds.core import MPJRDConfig


def test_stdp_uses_pre_stp_input_for_pre_spike_detection():
    if not pyfolds.ADVANCED_AVAILABLE:
        return
    cfg = MPJRDConfig(
        n_dendrites=1,
        n_synapses_per_dendrite=1,
        spike_threshold=0.5,
        stdp_input_source="raw",
    )
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)

    neuron.u_stp.fill_(0.1)
    neuron.R_stp.fill_(0.1)

    x_original = torch.ones(1, 1, 1)
    out = neuron.forward(x_original, dt=1.0)

    assert out["stdp_applied"].item() is True
    assert neuron.trace_pre[0, 0, 0].item() > 0.0


def test_stdp_ltd_depends_on_pre_spikes_not_post_spike_only():
    if not pyfolds.ADVANCED_AVAILABLE:
        return
    cfg = MPJRDConfig(n_dendrites=1, n_synapses_per_dendrite=1, plasticity_mode="stdp", ltd_rule="classic")
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)

    neuron._ensure_traces(1, torch.device("cpu"))
    neuron.trace_post.fill_(1.0)
    neuron.trace_pre.zero_()

    before = neuron.dendrites[0].synapses[0].I.item()
    x_no_pre = torch.zeros(1, 1, 1)
    neuron._update_stdp_traces(x_no_pre, torch.ones(1), dt=1.0)
    after = neuron.dendrites[0].synapses[0].I.item()

    assert after == before
