"""Benchmark simplificado de comparação de modelos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foldsnet.factory import create_foldsnet


BASELINES = {
    "LeNet": 60000,
    "MobileNet": 4200000,
    "ResNet": 11600000,
}


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    parser = argparse.ArgumentParser(description="Comparação de parâmetros da FOLDSNet")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    folds = create_foldsnet("4L", args.dataset)
    folds_params = _count_params(folds)

    print("Modelo,Parametros")
    print(f"FOLDSNet-4L,{folds_params}")
    for name, n_params in BASELINES.items():
        print(f"{name},{n_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
