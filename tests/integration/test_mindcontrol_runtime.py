"""Integration tests for MindControl runtime parameter injection."""

import pytest
import torch

import pyfolds


@pytest.mark.integration
@pytest.mark.parametrize("extreme_threshold", [10.0, 100.0])
def test_mindcontrol_runtime_injection_prevents_numerical_crash(extreme_threshold: float) -> None:
    """MindControl deve mutar parâmetros em runtime sem causar NaN/crash."""
    cfg = pyfolds.NeuronConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        defer_updates=True,
        theta_init=0.5,
        theta_min=0.1,
        theta_max=5.0,
        neuromod_mode="external",
        device="cpu",
    )
    neuron = pyfolds.MPJRDNeuron(cfg, enable_telemetry=True, telemetry_profile="heavy")
    neuron.set_mode(pyfolds.LearningMode.BATCH)

    def decision(event):
        if event.step_id == 3:
            return pyfolds.MutationCommand("activity_threshold", extreme_threshold)
        return None

    mind = pyfolds.MindControl(decision_fn=decision)
    mind.register_neuron(neuron)

    sink = pyfolds.DistributorSink([
        neuron.telemetry.sink,
        pyfolds.MindControlSink(mind),
    ])
    neuron.telemetry.sink = sink

    for step in range(10):
        x = torch.rand(16, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
        out = neuron.step(x, reward=0.25, collect_stats=True)

        assert torch.isfinite(out["u"]).all()
        assert torch.isfinite(out["spikes"]).all()

        neuron.apply_plasticity(reward=0.25)

        assert torch.isfinite(neuron.theta).all()
        assert torch.isfinite(neuron.r_hat).all()

    assert neuron.cfg.activity_threshold == pytest.approx(extreme_threshold)
    assert neuron.cfg.target_spike_rate == pytest.approx(cfg.target_spike_rate)
