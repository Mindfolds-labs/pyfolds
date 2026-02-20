from __future__ import annotations

import argparse
from datetime import datetime

from training.mnist_pipeline import TrainArgs, run_training


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Treino MNIST com backend folds")
    parser.add_argument("--model", choices=["py", "wave"], default="py")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--console", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default="train.log")
    parser.add_argument("--sheer-cmd", default="")
    ns = parser.parse_args()
    return TrainArgs(
        backend="folds",
        model=ns.model,
        epochs=ns.epochs,
        batch=ns.batch,
        lr=ns.lr,
        run_id=ns.run_id,
        resume=ns.resume,
        device=ns.device,
        console=ns.console,
        log_level=ns.log_level,
        log_file=ns.log_file,
        sheer_cmd=ns.sheer_cmd,
    )


if __name__ == "__main__":
    raise SystemExit(run_training(parse_args()))
