import torch
import pyfolds


def test_connectivity_and_pruning_masks_affect_dendritic_potentials():
    cfg = pyfolds.NeuronConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=3,
        pruning_enabled=True,
        pruning_strategy="static",
        pruning_runtime_threshold=0.0,
        device="cpu",
    )
    neuron = pyfolds.MPJRDNeuron(cfg)

    x = torch.ones(1, 2, 3)
    full = neuron._compute_dendritic_potentials_vectorized(x)

    with torch.no_grad():
        neuron.connectivity_mask[0, 0] = 0.0
        neuron.pruning_mask[1, 2] = 0.0
    masked = neuron._compute_dendritic_potentials_vectorized(x)

    assert not torch.allclose(full, masked)


def test_runtime_buffers_are_not_persistent_in_state_dict():
    cfg = pyfolds.NeuronConfig(n_dendrites=2, n_synapses_per_dendrite=4, device="cpu")
    neuron = pyfolds.MPJRDNeuron(cfg)
    keys = neuron.state_dict().keys()

    assert "connectivity_mask" in keys
    assert "pruning_mask" in keys
    assert "_trace_winner_idx" not in keys
    assert "_trace_signal" not in keys
    assert "_trace_ptr" not in keys
    assert "phase_activity_hist" not in keys


def test_collect_reports_expose_expected_observability_fields():
    cfg = pyfolds.NeuronConfig(n_dendrites=2, n_synapses_per_dendrite=4, device="cpu")
    neuron = pyfolds.MPJRDNeuron(cfg)
    _ = neuron(torch.rand(3, 2, 4), collect_stats=False)

    c = neuron.collect_connectivity_snapshot()
    p = neuron.collect_pruning_snapshot()
    phase = neuron.collect_phase_activity_report()
    engram = neuron.collect_engram_report()

    assert "effective_connectivity" in c
    assert "pruned_ratio" in p
    assert "delta_vs_baseline" in phase
    assert "resonance_by_dendrite" in engram
