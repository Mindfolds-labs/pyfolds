"""Tests for continuous neuron health monitoring."""

import pyfolds
import torch
from pyfolds.monitoring import NeuronHealthMonitor


def test_health_monitor_runs_and_scores():
    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4, device="cpu")
    neuron = pyfolds.MPJRDNeuron(cfg)
    monitor = NeuronHealthMonitor(neuron, check_every_n_steps=1)

    x = torch.zeros(2, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
    neuron.forward(x)
    issues = monitor.check_health()

    assert isinstance(issues, dict)
    assert monitor.get_health_score() <= 100.0
