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


def _int_to_bool(value: int) -> bool:
    return bool(int(value))


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Treino MNIST unificado (.fold/.mind)")
    parser.add_argument("--backend", choices=["folds", "mind", "both"], default="both")
    parser.add_argument("--model", choices=["mpjrd", "foldsnet"], default="mpjrd")
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

    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--save-fold", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save-mind", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save-pt", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save-log", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save-metrics", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save-summary", type=int, choices=[0, 1], default=1)

    ns = parser.parse_args()

    save_fold = _int_to_bool(ns.save_fold)
    save_mind = _int_to_bool(ns.save_mind)
    if ns.backend == "folds":
        save_fold, save_mind = True, False
    elif ns.backend == "mind":
        save_fold, save_mind = False, True

    return TrainArgs(
        backend="folds" if save_fold else "mind",
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
        timesteps=ns.timesteps,
        hidden=ns.hidden,
        threshold=ns.threshold,
        save_fold=save_fold,
        save_mind=save_mind,
        save_pt=_int_to_bool(ns.save_pt),
        save_log=_int_to_bool(ns.save_log),
        save_metrics=_int_to_bool(ns.save_metrics),
        save_summary=_int_to_bool(ns.save_summary),
    )


if __name__ == "__main__":
    raise SystemExit(run_training(parse_args()))
