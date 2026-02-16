#!/usr/bin/env python3
"""Executa benchmarks de serialização e publica artefatos em docs/."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization.foldio import load_fold_or_mind, save_fold_or_mind


@dataclass
class BenchmarkStats:
    avg_seconds: float
    min_seconds: float
    max_seconds: float
    ops_per_second: float
    mb_per_second: float


def _build_neuron(seed: int) -> MPJRDNeuron:
    torch.manual_seed(seed)
    cfg = MPJRDConfig(n_dendrites=8, n_synapses_per_dendrite=256, random_seed=seed)
    return MPJRDNeuron(cfg)


def _resolve_compression_mode() -> str:
    return "zstd" if importlib.util.find_spec("zstandard") is not None else "none"


def _measure_write_speed(
    neuron: MPJRDNeuron,
    iterations: int,
    workdir: Path,
    compression_mode: str,
) -> tuple[BenchmarkStats, int, int]:
    elapsed: list[float] = []
    sizes: list[int] = []

    for idx in range(iterations):
        path = workdir / f"write_{idx}.fold"
        start = time.perf_counter()
        save_fold_or_mind(neuron, str(path), compress=compression_mode)
        elapsed.append(time.perf_counter() - start)
        sizes.append(path.stat().st_size)

    avg_time = mean(elapsed)
    avg_size = int(mean(sizes))
    return (
        BenchmarkStats(
            avg_seconds=avg_time,
            min_seconds=min(elapsed),
            max_seconds=max(elapsed),
            ops_per_second=1.0 / avg_time,
            mb_per_second=(avg_size / (1024**2)) / avg_time,
        ),
        avg_size,
        sizes[-1],
    )


def _measure_read_speed(path: Path, iterations: int) -> BenchmarkStats:
    elapsed: list[float] = []
    file_size = path.stat().st_size

    for _ in range(iterations):
        start = time.perf_counter()
        _ = load_fold_or_mind(str(path), neuron_class=MPJRDNeuron, trusted_torch_payload=True)
        elapsed.append(time.perf_counter() - start)

    avg_time = mean(elapsed)
    return BenchmarkStats(
        avg_seconds=avg_time,
        min_seconds=min(elapsed),
        max_seconds=max(elapsed),
        ops_per_second=1.0 / avg_time,
        mb_per_second=(file_size / (1024**2)) / avg_time,
    )


def _measure_compression_ratio(neuron: MPJRDNeuron, workdir: Path, compression_mode: str) -> dict[str, Any]:
    raw_path = workdir / "reference_raw.fold"
    compressed_path = workdir / "reference_compressed.fold"

    save_fold_or_mind(neuron, str(raw_path), compress="none")

    method = "foldio-zstd"
    if compression_mode == "zstd":
        save_fold_or_mind(neuron, str(compressed_path), compress="zstd")
        compressed_size = compressed_path.stat().st_size
    else:
        method = "zlib-fallback"
        raw_bytes = raw_path.read_bytes()
        compressed_size = len(zlib.compress(raw_bytes, level=6))

    raw_size = raw_path.stat().st_size
    ratio = raw_size / compressed_size if compressed_size else 0.0

    return {
        "raw_size_bytes": raw_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": ratio,
        "space_saving_percent": (1.0 - (compressed_size / raw_size)) * 100.0,
        "method": method,
    }


def _generate_markdown(report: dict[str, Any]) -> str:
    ws = report["write_speed"]
    rs = report["read_speed"]
    cr = report["compression"]

    return "\n".join(
        [
            "# Benchmarks",
            "",
            "Relatório gerado automaticamente por `scripts/run_benchmarks.py`.",
            "",
            "## Resumo",
            "",
            f"- Data (UTC): `{report['generated_at_utc']}`",
            f"- Iterações: `{report['iterations']}`",
            f"- Python: `{report['environment']['python_version']}`",
            f"- PyTorch: `{report['environment']['torch_version']}`",
            f"- Plataforma: `{report['environment']['platform']}`",
            "",
            "## Velocidade de escrita",
            "",
            "| Métrica | Valor |",
            "|---|---:|",
            f"| Tempo médio | {ws['avg_seconds']:.6f} s |",
            f"| Tempo mínimo | {ws['min_seconds']:.6f} s |",
            f"| Tempo máximo | {ws['max_seconds']:.6f} s |",
            f"| Ops/s | {ws['ops_per_second']:.2f} |",
            f"| Throughput | {ws['mb_per_second']:.2f} MB/s |",
            "",
            "## Velocidade de leitura",
            "",
            "| Métrica | Valor |",
            "|---|---:|",
            f"| Tempo médio | {rs['avg_seconds']:.6f} s |",
            f"| Tempo mínimo | {rs['min_seconds']:.6f} s |",
            f"| Tempo máximo | {rs['max_seconds']:.6f} s |",
            f"| Ops/s | {rs['ops_per_second']:.2f} |",
            f"| Throughput | {rs['mb_per_second']:.2f} MB/s |",
            "",
            "## Compressão",
            "",
            "| Métrica | Valor |",
            "|---|---:|",
            f"| Tamanho sem compressão | {cr['raw_size_bytes']} bytes |",
            f"| Tamanho comprimido | {cr['compressed_size_bytes']} bytes |",
            f"| Razão de compressão (raw/compressed) | {cr['compression_ratio']:.3f}x |",
            f"| Economia de espaço | {cr['space_saving_percent']:.2f}% |",
            f"| Método | {cr['method']} |",
            "",
            "## Fonte",
            "",
            "- JSON completo: `docs/assets/benchmarks_results.json`",
        ]
    ) + "\n"


def run(iterations: int, output_json: Path, output_markdown: Path) -> dict[str, Any]:
    neuron = _build_neuron(seed=42)
    compression_mode = _resolve_compression_mode()

    with tempfile.TemporaryDirectory(prefix="pyfolds-bench-") as tmp:
        workdir = Path(tmp)
        write_stats, _, _ = _measure_write_speed(neuron, iterations, workdir, compression_mode)

        ref_path = workdir / "read_target.fold"
        save_fold_or_mind(neuron, str(ref_path), compress=compression_mode)
        read_stats = _measure_read_speed(ref_path, iterations)

        compression = _measure_compression_ratio(neuron, workdir, compression_mode)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "iterations": iterations,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        },
        "compression_mode": compression_mode,
        "write_speed": asdict(write_stats),
        "read_speed": asdict(read_stats),
        "compression": compression,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_generate_markdown(report), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa benchmarks de serialização do PyFolds")
    parser.add_argument("--iterations", type=int, default=20, help="Número de iterações por métrica")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/assets/benchmarks_results.json"),
        help="Arquivo JSON de saída",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("docs/BENCHMARKS.md"),
        help="Arquivo markdown de saída",
    )

    args = parser.parse_args()
    report = run(iterations=args.iterations, output_json=args.output_json, output_markdown=args.output_markdown)
    print(
        "Benchmarks concluídos:",
        f"write_ops_s={report['write_speed']['ops_per_second']:.2f}",
        f"read_ops_s={report['read_speed']['ops_per_second']:.2f}",
        f"compression={report['compression']['compression_ratio']:.2f}x",
    )


if __name__ == "__main__":
    main()
