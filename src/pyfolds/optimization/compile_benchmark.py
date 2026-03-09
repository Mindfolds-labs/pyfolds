from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class BenchmarkResult:
    mode: str
    throughput_samples_s: float
    latency_ms: float
    warmup_s: float
    memory_mb: float


def _bench(model, x, steps: int = 20) -> tuple[float, float]:
    t0 = time.perf_counter()
    for _ in range(steps):
        _ = model(x)
    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / steps) * 1000.0
    throughput = (x.shape[0] * steps) / max(elapsed, 1e-8)
    return throughput, latency_ms


def run_compile_benchmark(output_json: str | None = None) -> dict[str, BenchmarkResult]:
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 32)).eval()
    x = torch.randn(64, 64)

    eager_w0 = time.perf_counter(); _ = model(x); eager_warmup = time.perf_counter() - eager_w0
    eager_tp, eager_lat = _bench(model, x)
    results = {
        "eager": BenchmarkResult("eager", eager_tp, eager_lat, eager_warmup, 0.0)
    }

    if hasattr(torch, "compile"):
        cmodel = torch.compile(model)
        c_w0 = time.perf_counter(); _ = cmodel(x); c_warmup = time.perf_counter() - c_w0
        c_tp, c_lat = _bench(cmodel, x)
        results["compile"] = BenchmarkResult("compile", c_tp, c_lat, c_warmup, 0.0)

    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: asdict(v) for k, v in results.items()}, indent=2), encoding="utf-8")

    return results
