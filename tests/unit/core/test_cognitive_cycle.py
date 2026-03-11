"""Tests for cognitive cycle, circadian gates and neuromodulatory persistence."""

import torch
import pyfolds


def _make_cfg(**kwargs):
    base = dict(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        device="cpu",
        circadian_enabled=True,
        circadian_cycle_hours=0.001,
        defer_updates=False,
        plastic=True,
        neuromod_mode="external",
        theta_init=0.4,
        theta_min=0.1,
        theta_max=2.0,
    )
    base.update(kwargs)
    return pyfolds.NeuronConfig(**base)


def test_circadian_gates_and_state_are_exposed():
    neuron = pyfolds.MPJRDNeuron(_make_cfg())
    x = torch.ones(4, 2, 4)

    out = neuron.step(x, reward=0.8, dt=2.0)

    assert "network_state" in out
    assert 0.0 <= out["circadian_consolidation_gate"] <= 1.0
    assert out["effective_eta"] >= 0.0


def test_cycle_transitions_include_sleep_replay_and_recovery():
    neuron = pyfolds.MPJRDNeuron(_make_cfg(replay_interval_steps=1))
    x = torch.ones(4, 2, 4)

    for _ in range(10):
        neuron.step(x, reward=1.0, dt=10.0)

    valid = {
        "active",
        "focused_learning",
        "sleep_consolidation",
        "memory_replay",
        "homeostatic_recovery",
    }
    assert neuron.network_state.value in valid


def test_checkpoint_persists_circadian_and_state_buffers():
    neuron = pyfolds.MPJRDNeuron(_make_cfg())
    x = torch.ones(2, 2, 4)
    neuron.step(x, reward=0.5, dt=3.0)

    state = neuron.state_dict()
    other = pyfolds.MPJRDNeuron(_make_cfg())
    other.load_state_dict(state)

    assert torch.allclose(neuron.circadian_phase, other.circadian_phase)
    assert torch.allclose(neuron.global_time_ms, other.global_time_ms)
