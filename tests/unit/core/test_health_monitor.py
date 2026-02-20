"""Tests for continuous neuron health monitoring."""

import pyfolds
import torch
from pyfolds.monitoring import NeuronHealthMonitor, WeightIntegrityMonitor


def test_health_monitor_runs_and_scores():
    cfg = pyfolds.NeuronConfig(n_dendrites=2, n_synapses_per_dendrite=4, device="cpu")
    neuron = pyfolds.MPJRDNeuron(cfg)
    monitor = NeuronHealthMonitor(neuron, check_every_n_steps=1)

    x = torch.zeros(2, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
    neuron.forward(x)
    issues = monitor.check_health()

    assert isinstance(issues, dict)
    assert monitor.get_health_score() <= 100.0



def test_weight_integrity_monitor_detects_change():
    model = torch.nn.Linear(4, 3)
    monitor = WeightIntegrityMonitor(model, check_every_n_steps=1)

    first = monitor.check_integrity()
    assert first["checked"] is True
    assert first["ok"] is True

    with torch.no_grad():
        model.weight.add_(0.01)

    second = monitor.check_integrity()
    assert second["checked"] is True
    assert second["ok"] is False


def test_weight_integrity_monitor_respects_interval():
    model = torch.nn.Linear(2, 2)
    monitor = WeightIntegrityMonitor(model, check_every_n_steps=3)

    assert monitor.check_integrity()["checked"] is False
    assert monitor.check_integrity()["checked"] is False
    third = monitor.check_integrity()

    assert third["checked"] is True
    assert third["ok"] is True
