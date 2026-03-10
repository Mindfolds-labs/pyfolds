"""Benchmark regression detector.

Compares the latest benchmark run against the moving average of the previous
five runs and exits with a non-zero code when the slowdown exceeds 15%.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
from typing import Any


def _load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark history file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("Benchmark history must contain at least two runs")
    return data


def detect_regression(runs: list[dict[str, Any]], threshold: float = 0.15) -> tuple[bool, str]:
    latest = runs[-1]
    baseline_runs = runs[-6:-1]
    latest_score = float(latest["throughput"])
    baseline = [float(item["throughput"]) for item in baseline_runs]
    baseline_mean = statistics.mean(baseline)
    drop_ratio = (baseline_mean - latest_score) / baseline_mean

    message = (
        f"latest={latest_score:.4f}, baseline_mean={baseline_mean:.4f}, "
        f"drop={drop_ratio * 100:.2f}%"
    )
    return drop_ratio > threshold, message


def main() -> int:
    history_path = Path("benchmarks/history.json")
    runs = _load_runs(history_path)
    has_regression, message = detect_regression(runs)
    print(message)
    return 1 if has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
