import logging
import math
import tracemalloc

import pytest
import torch

from tests.unit.serialization.test_foldio import _build_neuron

try:
    import psutil
except Exception:  # pragma: no cover - opcional
    psutil = None


@pytest.mark.stress
def test_stress_100k_steps_no_nan_inf_and_memory_stable():
    logging.getLogger("pyfolds.neuron").setLevel(logging.ERROR)

    neuron = _build_neuron(enable_telemetry=False)
    x = torch.rand(1, neuron.cfg.n_dendrites, neuron.cfg.n_synapses_per_dendrite)

    tracemalloc.start()
    process = psutil.Process() if psutil is not None else None

    start_current, start_peak = tracemalloc.get_traced_memory()
    rss_before = process.memory_info().rss if process is not None else None

    for _ in range(100_000):
        out = neuron.forward(x, collect_stats=False)
        neuron.apply_plasticity(dt=1.0)

    end_current, end_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    finite_tensors = [neuron.N, neuron.I, neuron.W]

    if torch.is_tensor(out):
        finite_tensors.append(out)
    elif isinstance(out, dict):
        finite_tensors.extend(v for v in out.values() if torch.is_tensor(v))
    elif isinstance(out, (list, tuple)):
        finite_tensors.extend(v for v in out if torch.is_tensor(v))
    for optional_name in ("u", "R", "protection"):
        optional_tensor = getattr(neuron, optional_name, None)
        if optional_tensor is not None and torch.is_tensor(optional_tensor):
            finite_tensors.append(optional_tensor)

    for tensor in finite_tensors:
        assert torch.isfinite(tensor).all(), "Detectado NaN/Inf após stress de 100k steps"

    extra_current_mb = (end_current - start_current) / (1024 * 1024)
    extra_peak_mb = (end_peak - start_peak) / (1024 * 1024)
    assert extra_current_mb < 50, f"Possível leak detectado por tracemalloc: +{extra_current_mb:.2f}MB"
    assert extra_peak_mb < 250, f"Pico de memória suspeito no stress: +{extra_peak_mb:.2f}MB"

    if process is not None:
        rss_after = process.memory_info().rss
        rss_growth_mb = (rss_after - rss_before) / (1024 * 1024)
        assert math.isfinite(rss_growth_mb)
        assert rss_growth_mb < 300, f"Possível leak de RSS detectado via psutil: +{rss_growth_mb:.2f}MB"
