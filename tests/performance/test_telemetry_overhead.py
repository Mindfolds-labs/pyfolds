"""Performance tests focused on telemetry overhead in forward pass."""

import time

import pytest
import pyfolds
import torch


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_forward_loop(neuron, x: torch.Tensor, *, steps: int = 300, warmup: int = 30):
    """Run repeated forward calls and collect per-step duration in milliseconds."""
    for _ in range(warmup):
        neuron(x)

    timings_ms = []
    for _ in range(steps):
        _sync_if_cuda()
        start = time.perf_counter()
        out = neuron(x)
        _sync_if_cuda()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Safety checks mentioned in the report discussion.
        assert out["spikes"].shape[0] == x.shape[0]
        assert not torch.isnan(out["u"]).any()

        timings_ms.append(elapsed_ms)

    mean_ms = sum(timings_ms) / len(timings_ms)
    p95_idx = max(0, int(0.95 * len(timings_ms)) - 1)
    p95_ms = sorted(timings_ms)[p95_idx]
    return mean_ms, p95_ms, timings_ms


@pytest.mark.performance
@pytest.mark.slow
def test_forward_telemetry_ringbuffer_overhead(small_config):
    """Compare forward-pass timing with and without telemetry enabled."""
    batch_size = 16
    x = torch.randn(batch_size, 2, 4)

    neuron_no_telemetry = pyfolds.MPJRDNeuron(small_config, enable_telemetry=False)
    neuron_with_telemetry = pyfolds.MPJRDNeuron(
        small_config,
        enable_telemetry=True,
        telemetry_profile="heavy",
    )

    base_mean_ms, base_p95_ms, _ = _measure_forward_loop(neuron_no_telemetry, x)
    telem_mean_ms, telem_p95_ms, _ = _measure_forward_loop(neuron_with_telemetry, x)

    overhead_ms = telem_mean_ms - base_mean_ms
    overhead_pct = (overhead_ms / base_mean_ms) * 100 if base_mean_ms > 0 else float("inf")

    snapshot = neuron_with_telemetry.telemetry.snapshot()
    assert len(snapshot) > 0

    # Non-flaky guardrail: if this explodes, telemetry likely regressed badly.
    assert overhead_pct < 2000, (
        "Telemetry overhead unexpectedly high: "
        f"base_mean={base_mean_ms:.3f}ms, telem_mean={telem_mean_ms:.3f}ms, "
        f"base_p95={base_p95_ms:.3f}ms, telem_p95={telem_p95_ms:.3f}ms, "
        f"overhead={overhead_pct:.1f}%"
    )
