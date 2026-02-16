#!/usr/bin/env python3
"""Executa benchmarks de serialização do PyFolds e gera artefatos em JSON/Markdown."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pyfolds.serialization.foldio import FoldReader, FoldWriter  # noqa: E402


def _check_optional_deps() -> List[str]:
    missing = []
    try:
        import zstandard  # noqa: F401
    except Exception:
        missing.append("zstandard")

    try:
        import google_crc32c  # noqa: F401
    except Exception:
        missing.append("google-crc32c")

    return missing


def _build_payload(size_mb: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 32, size=size_mb * 1024 * 1024, dtype=np.uint8)
    return raw.tobytes()


def _bench_once(payload: bytes, output_file: Path, compress: str, zstd_level: int) -> Dict[str, Any]:
    start_write = time.perf_counter()
    with FoldWriter(str(output_file), compress=compress, zstd_level=zstd_level) as writer:
        writer.add_chunk("payload", "DATA", payload)
        writer.finalize({"source": "scripts/run_benchmarks.py"})
    write_seconds = time.perf_counter() - start_write

    start_read = time.perf_counter()
    with FoldReader(str(output_file), use_mmap=True) as reader:
        loaded = reader.read_chunk_bytes("payload", verify=True)
    read_seconds = time.perf_counter() - start_read

    if loaded != payload:
        raise RuntimeError("Falha de integridade: payload lido difere do payload gravado.")

    compressed_size_bytes = output_file.stat().st_size
    uncompressed_size_bytes = len(payload)

    return {
        "write_seconds": write_seconds,
        "read_seconds": read_seconds,
        "write_mb_s": uncompressed_size_bytes / (1024 * 1024) / write_seconds,
        "read_mb_s": uncompressed_size_bytes / (1024 * 1024) / read_seconds,
        "compressed_size_bytes": compressed_size_bytes,
        "uncompressed_size_bytes": uncompressed_size_bytes,
        "compress_ratio": compressed_size_bytes / uncompressed_size_bytes,
    }


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = [
        "write_seconds",
        "read_seconds",
        "write_mb_s",
        "read_mb_s",
        "compressed_size_bytes",
        "uncompressed_size_bytes",
        "compress_ratio",
    ]
    aggregated: Dict[str, float] = {}
    for key in keys:
        aggregated[key] = float(sum(item[key] for item in results) / len(results))
    return aggregated


def run_benchmarks(
    iterations: int,
    payload_size_mb: int,
    zstd_level: int,
    seed: int,
    require_optional_deps: bool,
) -> Dict[str, Any]:
    missing = _check_optional_deps()
    if missing and require_optional_deps:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            "Dependências opcionais ausentes para benchmark reproduzível: "
            f"{missing_list}. Instale com `pip install .[serialization]`."
        )

    compress_mode = "zstd" if "zstandard" not in missing else "none"
    payload = _build_payload(payload_size_mb, seed)
    runs = []

    with tempfile.TemporaryDirectory(prefix="pyfolds-bench-") as tmp_dir:
        output_file = Path(tmp_dir) / "sample.fold"
        for _ in range(iterations):
            run = _bench_once(
                payload,
                output_file=output_file,
                compress=compress_mode,
                zstd_level=zstd_level,
            )
            runs.append(run)

    summary = _aggregate(runs)
    now = datetime.now(timezone.utc)

    return {
        "collected_at_utc": now.isoformat(),
        "benchmark": {
            "iterations": iterations,
            "payload_size_mb": payload_size_mb,
            "zstd_level": zstd_level,
            "compression_mode": compress_mode,
            "seed": seed,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "optional_dependencies": {
                "zstandard": "installed" if "zstandard" not in missing else "missing",
                "google-crc32c": "installed" if "google-crc32c" not in missing else "missing",
            },
        },
        "results": {
            "avg_write_speed_mb_s": summary["write_mb_s"],
            "avg_read_speed_mb_s": summary["read_mb_s"],
            "avg_compress_ratio": summary["compress_ratio"],
            "avg_write_seconds": summary["write_seconds"],
            "avg_read_seconds": summary["read_seconds"],
            "avg_compressed_size_bytes": summary["compressed_size_bytes"],
            "avg_uncompressed_size_bytes": summary["uncompressed_size_bytes"],
        },
        "runs": runs,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    collected = report["collected_at_utc"]
    bench = report["benchmark"]
    env = report["environment"]
    results = report["results"]

    lines = [
        "# Benchmarks",
        "",
        f"Data da coleta (UTC): **{collected}**",
        "",
        "## Configuração",
        "",
        f"- Iterações: `{bench['iterations']}`",
        f"- Tamanho do payload: `{bench['payload_size_mb']} MB`",
        f"- Nível ZSTD: `{bench['zstd_level']}`",
        f"- Modo de compressão: `{bench['compression_mode']}`",
        f"- Seed: `{bench['seed']}`",
        f"- Python: `{env['python']}`",
        f"- Plataforma: `{env['platform']}`",
        "",
        "## Métricas",
        "",
        "| Métrica | Valor médio |",
        "|---|---:|",
        f"| Write speed | {results['avg_write_speed_mb_s']:.2f} MB/s |",
        f"| Read speed | {results['avg_read_speed_mb_s']:.2f} MB/s |",
        f"| Compress ratio | {results['avg_compress_ratio']:.4f} |",
        f"| Write time | {results['avg_write_seconds']:.4f} s |",
        f"| Read time | {results['avg_read_seconds']:.4f} s |",
        f"| Compressed size | {results['avg_compressed_size_bytes']:.0f} bytes |",
        f"| Uncompressed size | {results['avg_uncompressed_size_bytes']:.0f} bytes |",
        "",
        "## Dependências opcionais",
        "",
        f"- zstandard: `{env['optional_dependencies']['zstandard']}`",
        f"- google-crc32c: `{env['optional_dependencies']['google-crc32c']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--payload-size-mb", type=int, default=16)
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=Path("docs/assets/benchmarks_results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/BENCHMARKS.md"))
    parser.add_argument("--require-optional-deps", action="store_true")
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations deve ser maior que zero")
    if args.payload_size_mb <= 0:
        raise ValueError("--payload-size-mb deve ser maior que zero")

    report = run_benchmarks(
        iterations=args.iterations,
        payload_size_mb=args.payload_size_mb,
        zstd_level=args.zstd_level,
        seed=args.seed,
        require_optional_deps=args.require_optional_deps,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"Benchmark JSON salvo em: {args.output_json}")
    print(f"Benchmark Markdown salvo em: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
