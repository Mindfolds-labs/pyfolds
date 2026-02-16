import math
import tracemalloc

import pytest
import torch


try:
    import psutil
except Exception:  # pragma: no cover - opcional
    psutil = None


@pytest.mark.slow
@pytest.mark.stress
def test_100k_steps_stability_and_memory_signals(tmp_path):
    del tmp_path  # explicita uso de fixture temporária sem IO obrigatório

    state = torch.zeros(8, 8, dtype=torch.float32)
    drive = torch.full((8, 8), 0.001, dtype=torch.float32)

    tracemalloc.start()

    rss_start = None
    if psutil is not None:
        rss_start = psutil.Process().memory_info().rss

    for step in range(100_000):
        state = state * 0.999 + drive

        if step % 5_000 == 0:
            assert torch.isfinite(state).all(), f"Estado inválido no step {step}"

    assert torch.isfinite(state).all()
    assert not torch.isnan(state).any()
    assert not torch.isinf(state).any()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert math.isfinite(float(current))
    assert math.isfinite(float(peak))
    assert peak < 256 * 1024 * 1024, f"Pico de tracemalloc alto: {peak} bytes"

    if psutil is not None and rss_start is not None:
        rss_end = psutil.Process().memory_info().rss
        assert (rss_end - rss_start) < 200 * 1024 * 1024, (
            f"Crescimento de RSS sugere leak: start={rss_start}, end={rss_end}"
        )
