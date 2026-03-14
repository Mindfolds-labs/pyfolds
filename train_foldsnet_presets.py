from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.mnist_pipeline import TrainArgs, run_training


def _build_args(dataset: str, run_id: str, epochs: int, batch: int, lr: float, device: str, console: bool) -> TrainArgs:
    return TrainArgs(
        backend="folds",
        model="foldsnet",
        epochs=epochs,
        batch=batch,
        lr=lr,
        run_id=run_id,
        resume=False,
        device=device,
        console=console,
        log_level="INFO",
        log_file="train.log",
        save_fold=1,
        save_mind=1,
        save_pt=1,
        save_metrics=1,
        save_summary=1,
        foldsnet_variant="4L",
        foldsnet_dataset=dataset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preset de treino FOLDSNet para MNIST/CIFAR")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"], default="mnist")
    parser.add_argument("--run-both", action="store_true", help="Roda sequência mnist -> cifar10")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--console", action="store_true")
    return parser.parse_args()


def main() -> int:
    ns = parse_args()
    base_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    datasets = ["mnist", "cifar10"] if ns.run_both else [ns.dataset]
    for ds in datasets:
        run_id = f"foldsnet_{ds}_{base_ts}"
        print(
            f"Comando equivalente: python train_mnist_folds.py --epochs {ns.epochs} --batch {ns.batch} "
            f"--lr {ns.lr} --model foldsnet --foldsnet-variant 4L --foldsnet-dataset {ds} "
            f"--device {ns.device} --run-id {run_id}{' --console' if ns.console else ''}"
        )
        rc = run_training(_build_args(ds, run_id, ns.epochs, ns.batch, ns.lr, ns.device, ns.console))
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
