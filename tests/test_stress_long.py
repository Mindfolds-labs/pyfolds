import importlib.util
import tracemalloc

import numpy as np
import pytest
import torch

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron


HAS_PSUTIL = importlib.util.find_spec("psutil") is not None
if HAS_PSUTIL:
    import psutil


@pytest.mark.stress
@pytest.mark.slow
def test_long_run_100k_steps_without_nan_inf_and_optional_leak_collection():
    cfg = MPJRDConfig(
        n_dendrites=1,
        n_synapses_per_dendrite=2,
        device="cpu",
        defer_updates=True,
        plastic=True,
    )
    neuron = MPJRDNeuron(cfg)

    x = torch.full((1, cfg.n_dendrites, cfg.n_synapses_per_dendrite), 0.25)

    tracemalloc.start()
    rss_start = None
    process = None
    if HAS_PSUTIL:
        process = psutil.Process()
        rss_start = process.memory_info().rss

    with torch.no_grad():
        for _ in range(100_000):
            neuron.forward(x)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracked_tensors = [
        getattr(neuron, "N", None),
        getattr(neuron, "I", None),
        getattr(neuron, "u", None),
        getattr(neuron, "R", None),
        getattr(neuron, "theta", None),
        getattr(neuron, "r_hat", None),
    ]
    for tensor in tracked_tensors:
        if tensor is not None:
            assert torch.isfinite(tensor).all()

    assert peak > 0

    if HAS_PSUTIL and process is not None and rss_start is not None:
        rss_end = process.memory_info().rss
        # Coleta de leak opcional para telemetria simples de regressão.
        assert (rss_end - rss_start) < 250 * 1024 * 1024

        main
