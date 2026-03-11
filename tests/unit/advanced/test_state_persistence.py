"""State persistence tests for advanced neuron buffers."""

import io

import pytest
import torch
import pyfolds


def _make_active_neuron():
    cfg = pyfolds.NeuronConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=3,
        adaptation_enabled=True,
        backprop_delay=0.0,
        device="cpu",
    )
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)
    x = torch.ones(2, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
    neuron.forward(x, dt=1.0, collect_stats=False)
    neuron.forward(x, dt=1.0, collect_stats=False)
    return neuron, cfg


def test_state_dict_completeness_for_buffers():
    if not pyfolds.ADVANCED_AVAILABLE:
        pytest.skip("Advanced module not available")

    neuron, _ = _make_active_neuron()
    state = neuron.state_dict()

    assert "last_spike_time" in state
    assert "adaptation_current" in state
    assert "backprop_trace" in state
    assert "dendrite_amplification" in state


def test_state_dict_round_trip_preserves_advanced_buffers():
    if not pyfolds.ADVANCED_AVAILABLE:
        pytest.skip("Advanced module not available")

    neuron, cfg = _make_active_neuron()

    snapshot = {
        "last_spike_time": neuron.last_spike_time.clone(),
        "adaptation_current": neuron.adaptation_current.clone(),
        "backprop_trace": neuron.backprop_trace.clone(),
        "dendrite_amplification": neuron.dendrite_amplification.clone(),
    }

    buffer = io.BytesIO()
    torch.save(neuron.state_dict(), buffer)
    buffer.seek(0)

    loaded = pyfolds.MPJRDNeuronAdvanced(cfg)
    loaded.load_state_dict(torch.load(buffer, map_location="cpu"))

    assert torch.allclose(loaded.last_spike_time, snapshot["last_spike_time"])
    assert torch.allclose(loaded.adaptation_current, snapshot["adaptation_current"])
    assert torch.allclose(loaded.backprop_trace, snapshot["backprop_trace"])
    assert torch.allclose(loaded.dendrite_amplification, snapshot["dendrite_amplification"])


def test_to_device_preserves_advanced_state_buffers():
    if not pyfolds.ADVANCED_AVAILABLE:
        pytest.skip("Advanced module not available")

    neuron, _ = _make_active_neuron()
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    before = {
        "last_spike_time": neuron.last_spike_time.clone(),
        "adaptation_current": neuron.adaptation_current.clone(),
        "backprop_trace": neuron.backprop_trace.clone(),
        "dendrite_amplification": neuron.dendrite_amplification.clone(),
    }

    neuron = neuron.to(target_device)

    assert neuron.last_spike_time.device == target_device
    assert neuron.adaptation_current.device == target_device
    assert neuron.backprop_trace.device == target_device
    assert neuron.dendrite_amplification.device == target_device

    assert torch.allclose(neuron.last_spike_time.cpu(), before["last_spike_time"])
    assert torch.allclose(neuron.adaptation_current.cpu(), before["adaptation_current"])
    assert torch.allclose(neuron.backprop_trace.cpu(), before["backprop_trace"])
    assert torch.allclose(neuron.dendrite_amplification.cpu(), before["dendrite_amplification"])
