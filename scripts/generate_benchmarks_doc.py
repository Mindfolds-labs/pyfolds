#!/usr/bin/env python3
"""Gera docs/BENCHMARKS.md a partir de docs/assets/benchmarks_results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def render(results: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Benchmarks de Serialização (FoldIO)")
    lines.append("")
    lines.append(f"- Gerado em: `{results['generated_at_utc']}`")
    lines.append(f"- Seed: `{results['seed']}`")
    lines.append(f"- Amostras por medição: `{results['sample_count']}`")
    lines.append(f"- Device: `{results['device']}`")
    lines.append(f"- Python: `{results['python_version']}`")
    lines.append(f"- PyTorch: `{results['torch_version']}`")
    lines.append(f"- Compressão Fold/ZSTD disponível: `{results['fold_zstd_available']}`")
    lines.append("")

    lines.append("## Throughput de escrita")
    lines.append("")
    if results["fold_zstd_available"]:
        lines.append("| Cenário | none (MiB/s) | zstd (MiB/s) | Arquivo none (bytes) | Arquivo zstd (bytes) |")
        lines.append("|---|---:|---:|---:|---:|")
    else:
        lines.append("| Cenário | none (MiB/s) | Arquivo none (bytes) |")
        lines.append("|---|---:|---:|")

    for row in results["benchmarks"]:
        sc = row["scenario"]
        if results["fold_zstd_available"]:
            lines.append(
                "| {name} ({d}x{s}, batch={b}) | {wn} | {wz} | {sn} | {sz} |".format(
                    name=sc["name"],
                    d=sc["n_dendrites"],
                    s=sc["n_synapses_per_dendrite"],
                    b=sc["batch_size"],
                    wn=_fmt(row["write"]["none"]["throughput_mib_per_s"]),
                    wz=_fmt(row["write"]["zstd"]["throughput_mib_per_s"]),
                    sn=row["write"]["none"]["median_bytes"],
                    sz=row["write"]["zstd"]["median_bytes"],
                )
            )
        else:
            lines.append(
                "| {name} ({d}x{s}, batch={b}) | {wn} | {sn} |".format(
                    name=sc["name"],
                    d=sc["n_dendrites"],
                    s=sc["n_synapses_per_dendrite"],
                    b=sc["batch_size"],
                    wn=_fmt(row["write"]["none"]["throughput_mib_per_s"]),
                    sn=row["write"]["none"]["median_bytes"],
                )
            )

    lines.append("")
    lines.append("## Throughput de leitura")
    lines.append("")
    if results["fold_zstd_available"]:
        lines.append("| Cenário | none (MiB/s) | zstd (MiB/s) |")
        lines.append("|---|---:|---:|")
    else:
        lines.append("| Cenário | none (MiB/s) |")
        lines.append("|---|---:|")

    for row in results["benchmarks"]:
        sc = row["scenario"]
        if results["fold_zstd_available"]:
            lines.append(
                "| {name} ({d}x{s}, batch={b}) | {rn} | {rz} |".format(
                    name=sc["name"],
                    d=sc["n_dendrites"],
                    s=sc["n_synapses_per_dendrite"],
                    b=sc["batch_size"],
                    rn=_fmt(row["read"]["none"]["throughput_mib_per_s"]),
                    rz=_fmt(row["read"]["zstd"]["throughput_mib_per_s"]),
                )
            )
        else:
            lines.append(
                "| {name} ({d}x{s}, batch={b}) | {rn} |".format(
                    name=sc["name"],
                    d=sc["n_dendrites"],
                    s=sc["n_synapses_per_dendrite"],
                    b=sc["batch_size"],
                    rn=_fmt(row["read"]["none"]["throughput_mib_per_s"]),
                )
            )

    lines.append("")
    lines.append("## Taxa de compressão")
    lines.append("")
    lines.append("| Cenário | Método | Razão vs none | Redução de espaço (%) |")
    lines.append("|---|---|---:|---:|")
    for row in results["benchmarks"]:
        sc = row["scenario"]
        lines.append(
            "| {name} | {method} | {ratio} | {reduction} |".format(
                name=sc["name"],
                method=row["compression"]["method"],
                ratio=_fmt(row["compression"]["ratio_vs_none"]),
                reduction=_fmt(row["compression"]["space_reduction_percent"]),
            )
        )

    lines.append("")
    lines.append(
        "Interpretação rápida: throughput maior é melhor; razão de compressão menor que 1.0 indica arquivo comprimido menor que o baseline `none`."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera docs/BENCHMARKS.md a partir do JSON")
    parser.add_argument("--input", type=Path, default=Path("docs/assets/benchmarks_results.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/BENCHMARKS.md"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.write_text(render(data), encoding="utf-8")


if __name__ == "__main__":
    main()
