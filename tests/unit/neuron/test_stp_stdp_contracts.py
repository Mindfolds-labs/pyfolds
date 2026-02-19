import torch
import pyfolds


def test_stdp_reads_pre_stp_input_for_pre_spikes_and_ltd_uses_pre_spike_gate():
    cfg = pyfolds.MPJRDConfig(
        n_dendrites=1,
        n_synapses_per_dendrite=1,
        plasticity_mode="stdp",
        stdp_trace_threshold=0.0,
        spike_threshold=0.5,
        A_plus=0.0,
        A_minus=0.5,
    )
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)

    neuron.u_stp.fill_(0.1)
    neuron.R_stp.fill_(0.1)
    neuron._ensure_traces(1, torch.device("cpu"))
    neuron.trace_post.fill_(1.0)

    before = neuron.dendrites[0].synapses[0].I.clone()
    x_original = torch.tensor([[[1.0]]])

    out = neuron.forward(x_original, mode=pyfolds.LearningMode.ONLINE)
    after = neuron.dendrites[0].synapses[0].I.clone()

    assert out["spikes"].shape == (1,)
    assert (after - before).item() < 0.0
