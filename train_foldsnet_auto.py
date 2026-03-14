from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.mnist_pipeline import TrainArgs, run_training


VALID_DATASETS = ("mnist", "cifar10", "cifar100")


def _resolve_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(VALID_DATASETS)
    if dataset_arg not in VALID_DATASETS:
        raise ValueError(f"dataset inválido: {dataset_arg}. Use {', '.join(VALID_DATASETS)} ou all")
    return [dataset_arg]


def _resolve_device(device_arg: str) -> str:
    if device_arg in {"cpu", "cuda"}:
        return device_arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _build_run_id(prefix: str, dataset: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{dataset}_{ts}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino rápido FOLDSNet (VSCode-friendly) para MNIST/CIFAR10/CIFAR100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="mnist", help="mnist, cifar10, cifar100 ou all")
    parser.add_argument("--epochs", type=int, default=10, help="Épocas por dataset")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--variant", choices=["2L", "4L", "5L", "6L"], default="4L", help="Variante FOLDSNet")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Dispositivo de treino")
    parser.add_argument("--run-prefix", default="foldsnet_auto", help="Prefixo do run_id")
    parser.add_argument("--resume", action="store_true", help="Resume do checkpoint.pt no mesmo run-id")
    parser.add_argument("--console", action="store_true", help="Mostrar logs em console")
    parser.add_argument("--synthetic", action="store_true", help="Usa dataset sintético para debug rápido")
    parser.add_argument("--synthetic-train-samples", type=int, default=2048)
    parser.add_argument("--synthetic-test-samples", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    ns = _parse_args()
    device = _resolve_device(ns.device)
    datasets = _resolve_datasets(ns.dataset)

    if ns.synthetic:
        os.environ["PYFOLDS_MNIST_SYNTHETIC"] = "1"
        os.environ["PYFOLDS_SYNTH_TRAIN_SAMPLES"] = str(ns.synthetic_train_samples)
        os.environ["PYFOLDS_SYNTH_TEST_SAMPLES"] = str(ns.synthetic_test_samples)

    print("=" * 88)
    print("FOLDSNet auto trainer")
    print(f"datasets={datasets} | epochs={ns.epochs} | batch={ns.batch} | lr={ns.lr} | variant={ns.variant}")
    print(f"device={device} | synthetic={ns.synthetic} | resume={ns.resume}")
    print("Obs: mecanismo LeibReg/Proximity é benchmark separado e não altera este pipeline de treino.")
    print("=" * 88)

    exit_code = 0
    for ds in datasets:
        run_id = _build_run_id(ns.run_prefix, ds)
        args = TrainArgs(
            backend="folds",
            model="foldsnet",
            epochs=ns.epochs,
            batch=ns.batch,
            lr=ns.lr,
            run_id=run_id,
            resume=ns.resume,
            device=device,
            console=ns.console,
            log_level="INFO",
            log_file="train.log",
            foldsnet_variant=ns.variant,
            foldsnet_dataset=ds,
            save_fold=1,
            save_mind=0,
            save_pt=1,
            save_log=1,
            save_metrics=1,
            save_summary=1,
        )

        print(f"\n[START] dataset={ds} run_id={run_id}")
        print(f"[CONFIG] {asdict(args)}")
        code = run_training(args)
        print(f"[DONE] dataset={ds} exit_code={code} artifacts_dir=runs/{run_id}")
        if code != 0:
            exit_code = code

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
