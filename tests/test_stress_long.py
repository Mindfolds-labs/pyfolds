import numpy as np
import pytest
import torch

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron


@pytest.mark.stress
def test_long_run_100k_steps_without_nan_or_inf():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        device="cpu",
        defer_updates=True,
        plastic=True,
    )
    neuron = MPJRDNeuron(cfg)
    neuron.logger.setLevel("ERROR")

    for _ in range(100_000):
        x = torch.rand(1, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
        neuron.forward(x, collect_stats=False)

    neuron.apply_plasticity(dt=1.0)

    monitored_arrays = {
        "N": neuron.N.detach().cpu().numpy(),
        "I": neuron.I.detach().cpu().numpy(),
        "W": neuron.W.detach().cpu().numpy(),
        "theta": np.array([neuron.theta.item()]),
        "r_hat": np.array([neuron.r_hat.item()]),
    }

    for name, arr in monitored_arrays.items():
        assert np.isfinite(arr).all(), f"Array {name} contém NaN/Inf após 100k steps"
