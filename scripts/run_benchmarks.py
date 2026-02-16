#!/usr/bin/env python3
"""Executa benchmarks de serialização FoldIO e exporta JSON determinístico."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from pyfolds import MPJRDConfig, MPJRDNeuron, load_fold_or_mind, save_fold_or_mind

SEED = 1337


@dataclass(frozen=True)
class Scenario:
    name: str
    n_dendrites: int
    n_synapses_per_dendrite: int
    batch_size: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_neuron(scenario: Scenario) -> MPJRDNeuron:
    cfg = MPJRDConfig(
        n_dendrites=scenario.n_dendrites,
        n_synapses_per_dendrite=scenario.n_synapses_per_dendrite,
        random_seed=SEED,
    )
    neuron = MPJRDNeuron(cfg)
    x = torch.randn(scenario.batch_size, scenario.n_dendrites, scenario.n_synapses_per_dendrite)
    neuron(x, reward=0.15)
    return neuron


def _measure_write_throughput(neuron: MPJRDNeuron, scenario: Scenario, compress: str, sample_count: int) -> Dict[str, float]:
    samples_seconds: List[float] = []
    sample_sizes: List[int] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        for idx in range(sample_count):
            out_path = base / f"{scenario.name}-{compress}-{idx}.fold"
            t0 = time.perf_counter()
            save_fold_or_mind(
                neuron,
                str(out_path),
                include_history=False,
                include_telemetry=False,
                include_nuclear_arrays=True,
                compress=compress,
            )
            dt = max(time.perf_counter() - t0, 1e-9)
            samples_seconds.append(dt)
            sample_sizes.append(out_path.stat().st_size)

    median_seconds = statistics.median(samples_seconds)
    median_size = int(statistics.median(sample_sizes))
    throughput_mib_s = (median_size / (1024 * 1024)) / median_seconds
    return {
        "median_seconds": round(median_seconds, 6),
        "median_bytes": median_size,
        "throughput_mib_per_s": round(throughput_mib_s, 6),
    }


def _measure_read_throughput(scenario: Scenario, compress: str, sample_count: int) -> Dict[str, float]:
    samples_seconds: List[float] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / f"{scenario.name}-{compress}.fold"
        neuron = _build_neuron(scenario)
        save_fold_or_mind(
            neuron,
            str(target),
            include_history=False,
            include_telemetry=False,
            include_nuclear_arrays=True,
            compress=compress,
        )
        file_size = target.stat().st_size

        for _ in range(sample_count):
            t0 = time.perf_counter()
            _ = load_fold_or_mind(str(target), MPJRDNeuron)
            samples_seconds.append(max(time.perf_counter() - t0, 1e-9))

    median_seconds = statistics.median(samples_seconds)
    throughput_mib_s = (file_size / (1024 * 1024)) / median_seconds
    return {
        "median_seconds": round(median_seconds, 6),
        "median_bytes": file_size,
        "throughput_mib_per_s": round(throughput_mib_s, 6),
    }


def _zlib_compression_ratio(neuron: MPJRDNeuron, scenario: Scenario) -> Dict[str, float]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{scenario.name}-none.fold"
        save_fold_or_mind(
            neuron,
            str(path),
            include_history=False,
            include_telemetry=False,
            include_nuclear_arrays=True,
            compress="none",
        )
        raw = path.read_bytes()

    compressed = zlib.compress(raw, level=6)
    ratio = len(compressed) / max(len(raw), 1)
    return {
        "method": "zlib(level=6)",
        "ratio_vs_none": round(ratio, 6),
        "space_reduction_percent": round((1.0 - ratio) * 100.0, 4),
    }


def _benchmark_scenario(scenario: Scenario, sample_count: int, include_zstd: bool) -> Dict[str, object]:
    _set_seed(SEED)
    neuron = _build_neuron(scenario)

    write = {"none": _measure_write_throughput(neuron, scenario, compress="none", sample_count=sample_count)}
    read = {"none": _measure_read_throughput(scenario, compress="none", sample_count=sample_count)}

    if include_zstd:
        write["zstd"] = _measure_write_throughput(neuron, scenario, compress="zstd", sample_count=sample_count)
        read["zstd"] = _measure_read_throughput(scenario, compress="zstd", sample_count=sample_count)
        ratio = write["zstd"]["median_bytes"] / max(write["none"]["median_bytes"], 1)
        compression = {
            "method": "fold:zstd",
            "ratio_vs_none": round(ratio, 6),
            "space_reduction_percent": round((1.0 - ratio) * 100.0, 4),
        }
    else:
        compression = _zlib_compression_ratio(neuron, scenario)

    return {
        "scenario": asdict(scenario),
        "write": write,
        "read": read,
        "compression": compression,
    }


def run(output_path: Path, sample_count: int) -> Dict[str, object]:
    scenarios = [
        Scenario(name="small", n_dendrites=4, n_synapses_per_dendrite=16, batch_size=16),
        Scenario(name="medium", n_dendrites=8, n_synapses_per_dendrite=32, batch_size=32),
    ]
    include_zstd = importlib.util.find_spec("zstandard") is not None
    rows = [_benchmark_scenario(s, sample_count=sample_count, include_zstd=include_zstd) for s in scenarios]

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "seed": SEED,
        "sample_count": sample_count,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "fold_zstd_available": include_zstd,
        "benchmarks": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa benchmarks de serialização do PyFolds")
    parser.add_argument("--output", type=Path, default=Path("docs/assets/benchmarks_results.json"))
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples deve ser >= 1")

    run(output_path=args.output, sample_count=args.samples)


if __name__ == "__main__":
    main()
