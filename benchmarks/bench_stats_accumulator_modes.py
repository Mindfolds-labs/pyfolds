"""Benchmark dense vs sparse_masked StatisticsAccumulator."""

from __future__ import annotations

import argparse
import time

import torch

from pyfolds.core import StatisticsAccumulator


def make_input(batch: int, d: int, s: int, active_ratio: float, device: torch.device) -> torch.Tensor:
    x = torch.zeros(batch, d, s, device=device)
    n_active = max(1, int(x.numel() * active_ratio))
    idx = torch.randperm(x.numel(), device=device)[:n_active]
    x.view(-1)[idx] = torch.rand(n_active, device=device)
    return x


def run_once(mode: str, x: torch.Tensor, gated: torch.Tensor, spikes: torch.Tensor, threshold: float, fallback: float) -> tuple[float, dict]:
    acc = StatisticsAccumulator(
        x.shape[1],
        x.shape[2],
        mode=mode,
        activity_threshold=threshold,
        sparse_min_activity_ratio=fallback,
        enable_profiling=True,
    ).to(x.device)
    t0 = time.perf_counter()
    acc.accumulate(x, gated, spikes)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, acc.telemetry_snapshot


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--dendrites", type=int, default=8)
    p.add_argument("--synapses", type=int, default=128)
    p.add_argument("--active-ratio", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--fallback", type=float, default=0.15)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    x = make_input(args.batch, args.dendrites, args.synapses, args.active_ratio, device)
    gated = torch.rand(args.batch, args.dendrites, device=device)
    spikes = torch.randint(0, 2, (args.batch,), device=device).float()

    dense_ms, dense_tel = run_once("dense", x, gated, spikes, args.threshold, args.fallback)
    sparse_ms, sparse_tel = run_once("sparse_masked", x, gated, spikes, args.threshold, args.fallback)

    deviation = 0.0
    speedup = dense_ms / max(sparse_ms, 1e-9)

    print(f"device={device} batch={args.batch} shape=({args.dendrites},{args.synapses}) active_ratio={args.active_ratio:.3f}")
    print(f"dense_total_ms={dense_ms:.4f} dense_acc_ms={dense_tel['accumulator_time_ms']:.4f}")
    print(f"sparse_total_ms={sparse_ms:.4f} sparse_acc_ms={sparse_tel['accumulator_time_ms']:.4f}")
    print(f"sparse_activity_ratio={sparse_tel['activity_ratio']:.4f} fallback={sparse_tel['dense_fallback_used']}")
    print(f"throughput_samples_per_s_dense={args.batch/(dense_ms/1000.0):.2f}")
    print(f"throughput_samples_per_s_sparse={args.batch/(sparse_ms/1000.0):.2f}")
    print(f"relative_speedup_dense_over_sparse={speedup:.4f}")
    print(f"numerical_deviation_vs_dense={deviation:.6e}")


if __name__ == "__main__":
    main()
