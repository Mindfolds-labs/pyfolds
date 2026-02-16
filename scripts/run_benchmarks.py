#!/usr/bin/env python3
"""Run deterministic serialization benchmarks and emit JSON + Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pyfolds.serialization.foldio import FoldReader, FoldWriter, zstd

DEFAULT_OUTPUT = Path("docs/assets/benchmarks_results.json")
DEFAULT_DOC = Path("docs/BENCHMARKS.md")
PAYLOAD_SIZE_MB = 16
PAYLOAD_CHUNK_NAME = "benchmark_payload"
PAYLOAD_CTYPE = "BENC"


@dataclass(frozen=True)
class BenchmarkResult:
    codec: str
    payload_size_bytes: int
    file_size_bytes: int
    write_speed_mb_s: float
    read_speed_mb_s: float
    compress_ratio: float


def _build_payload(size_bytes: int) -> bytes:
    # Deterministic pseudo-random payload using repeated SHA256 digest expansion.
    seed = b"pyfolds-benchmark-payload-v1"
    out = bytearray()
    counter = 0
    while len(out) < size_bytes:
        digest = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        out.extend(digest)
        counter += 1
    return bytes(out[:size_bytes])


def _measure_once(codec: str, payload: bytes, workdir: Path) -> BenchmarkResult:
    bench_path = workdir / f"benchmark_{codec}.fold"

    write_start = time.perf_counter()
    with FoldWriter(str(bench_path), compress=codec) as writer:
        writer.add_chunk(PAYLOAD_CHUNK_NAME, PAYLOAD_CTYPE, payload)
        writer.finalize(metadata={"benchmark": True, "codec": codec})
    write_seconds = max(time.perf_counter() - write_start, 1e-9)

    read_start = time.perf_counter()
    with FoldReader(str(bench_path), use_mmap=False) as reader:
        restored = reader.read_chunk_bytes(PAYLOAD_CHUNK_NAME)
    read_seconds = max(time.perf_counter() - read_start, 1e-9)

    if restored != payload:
        raise RuntimeError(f"Invalid roundtrip for codec={codec}")

    payload_size = len(payload)
    file_size = bench_path.stat().st_size
    payload_mb = payload_size / (1024 * 1024)

    return BenchmarkResult(
        codec=codec,
        payload_size_bytes=payload_size,
        file_size_bytes=file_size,
        write_speed_mb_s=round(payload_mb / write_seconds, 4),
        read_speed_mb_s=round(payload_mb / read_seconds, 4),
        compress_ratio=round(payload_size / file_size, 6),
    )


def _run_codec(codec: str, payload: bytes, iterations: int, workdir: Path) -> BenchmarkResult:
    runs = [_measure_once(codec, payload, workdir) for _ in range(iterations)]

    return BenchmarkResult(
        codec=codec,
        payload_size_bytes=runs[0].payload_size_bytes,
        file_size_bytes=runs[-1].file_size_bytes,
        write_speed_mb_s=round(statistics.median(r.write_speed_mb_s for r in runs), 4),
        read_speed_mb_s=round(statistics.median(r.read_speed_mb_s for r in runs), 4),
        compress_ratio=round(statistics.median(r.compress_ratio for r in runs), 6),
    )


def _as_dict(result: BenchmarkResult) -> dict[str, Any]:
    return {
        "codec": result.codec,
        "payload_size_bytes": result.payload_size_bytes,
        "file_size_bytes": result.file_size_bytes,
        "write_speed_mb_s": result.write_speed_mb_s,
        "read_speed_mb_s": result.read_speed_mb_s,
        "compress_ratio": result.compress_ratio,
    }


def _render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Benchmarks",
        "",
        "Resultados automáticos gerados por `scripts/run_benchmarks.py`.",
        "",
        f"- Tamanho do payload: `{data['config']['payload_size_mb']} MB`",
        f"- Iterações por codec: `{data['config']['iterations']}`",
        f"- Codecs avaliados: `{', '.join(data['config']['codecs'])}`",
        "",
        "| Codec | Write speed (MB/s) | Read speed (MB/s) | Compress ratio | Payload size (bytes) | File size (bytes) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for result in data["results"]:
        lines.append(
            "| {codec} | {write_speed_mb_s:.4f} | {read_speed_mb_s:.4f} | {compress_ratio:.6f} | {payload_size_bytes} | {file_size_bytes} |".format(
                **result
            )
        )

    lines.extend(
        [
            "",
            "## Reprodução local",
            "",
            "```bash",
            "python scripts/run_benchmarks.py",
            "```",
            "",
            "Este comando atualiza:",
            "- `docs/assets/benchmarks_results.json`",
            "- `docs/BENCHMARKS.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    payload = _build_payload(PAYLOAD_SIZE_MB * 1024 * 1024)
    codecs = ["none"]
    if zstd is not None:
        codecs.append("zstd")

    with tempfile.TemporaryDirectory(prefix="pyfolds-bench-") as tmp:
        workdir = Path(tmp)
        results = [_as_dict(_run_codec(codec, payload, args.iterations, workdir)) for codec in codecs]

    output_data = {
        "benchmark_version": 1,
        "config": {
            "payload_size_mb": PAYLOAD_SIZE_MB,
            "iterations": args.iterations,
            "codecs": codecs,
        },
        "results": sorted(results, key=lambda item: item["codec"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(_render_markdown(output_data), encoding="utf-8")

    print(f"Benchmark JSON written to: {args.output}")
    print(f"Benchmark report written to: {args.doc}")


if __name__ == "__main__":
    main()
