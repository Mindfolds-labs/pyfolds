"""CLI de treino para MNIST/CIFAR com FOLDSNet/MPJRD.

Exemplos:
  python train_mnist_folds.py --epochs 10 --batch 64 --lr 1e-3 --model foldsnet
  python train_mnist_folds.py --epochs 10 --batch 64 --lr 1e-3 --model mpjrd --hidden 128
"""

from __future__ import annotations

import json
import secrets
import string

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.mnist_pipeline import TrainArgs, run_training


def _generate_short_id(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _default_run_id(model: str, dataset: str) -> str:
    return f"{model}_{dataset}_{_generate_short_id()}"


def _save_run_metadata(args: TrainArgs) -> None:
    run_dir = Path("runs") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "run_id": args.run_id,
        "model": args.model,
        "dataset": args.foldsnet_dataset,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "device": args.device,
        "init_checkpoint": args.init_checkpoint,
        "config": args.__dict__,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treino MNIST com backend folds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    obrigatorios = parser.add_argument_group("obrigatórios")
    obrigatorios.add_argument("--epochs", type=int, required=True)
    obrigatorios.add_argument("--batch", "--batch-size", dest="batch", type=int, required=True)
    obrigatorios.add_argument("--lr", type=float, required=True)

    identificacao = parser.add_argument_group("identificação do run")
    identificacao.add_argument("--run-id", default="")
    identificacao.add_argument("--resume", action="store_true")
    identificacao.add_argument("--init-checkpoint", default="", help="Checkpoint .pt para warm start (carrega pesos)")

    hardware = parser.add_argument_group("hardware")
    hardware.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    modelo = parser.add_argument_group("modelo")
    modelo.add_argument("--model", choices=["mpjrd", "foldsnet"], default="foldsnet")
    modelo.add_argument("--foldsnet-variant", default="4L")
    modelo.add_argument("--foldsnet-dataset", default="mnist")

    arquitetura = parser.add_argument_group("arquitetura MPJRD")
    arquitetura.add_argument("--n-dendrites", type=int, default=4, help="Número de dendritos")
    arquitetura.add_argument("--n-synapses", type=int, default=32, dest="n_synapses_per_dendrite", help="Sinapses por dendrito")
    arquitetura.add_argument("--hidden", type=int, default=128, help="Número de neurônios excitatórios")
    arquitetura.add_argument("--threshold", type=float, default=0.45, help="Limiar inicial de disparo")

    mecanismos = parser.add_argument_group("mecanismos biológicos")
    mecanismos.add_argument("--disable-stdp", action="store_true", help="Desativa STDP")
    mecanismos.add_argument("--disable-homeostase", action="store_true", help="Desativa Homeostase")
    mecanismos.add_argument("--disable-inibicao", action="store_true", help="Desativa Inibição")
    mecanismos.add_argument("--disable-refratario", action="store_true", help="Desativa Período refratário")
    mecanismos.add_argument("--disable-backprop", action="store_true", help="Desativa Backpropagação")
    mecanismos.add_argument("--disable-sfa", action="store_true", help="Desativa Adaptação SFA")
    mecanismos.add_argument("--disable-stp", action="store_true", help="Desativa Dinâmica STP")
    mecanismos.add_argument("--disable-wave", action="store_true", help="Desativa mecanismo Wave")
    mecanismos.add_argument("--disable-circadian", action="store_true", help="Desativa mecanismo Circadiano")
    mecanismos.add_argument("--disable-engram", action="store_true", help="Desativa memória por Engrams")
    mecanismos.add_argument("--disable-speech", action="store_true", help="Desativa Speech tracking")

    log_console = parser.add_argument_group("log e console")
    log_console.add_argument("--console", action="store_true")
    log_console.add_argument("--log-level", default="INFO")
    log_console.add_argument("--log-file", default="train.log")

    artefatos = parser.add_argument_group("artefatos de saída")
    artefatos.add_argument("--save-fold", type=int, default=1, help="Salvar .fold (0/1)")
    artefatos.add_argument("--save-mind", type=int, default=1, help="Salvar .mind (0/1)")
    artefatos.add_argument("--save-pt", type=int, default=1, help="Salvar .pt (0/1)")
    artefatos.add_argument("--save-log", type=int, default=1, help="Salvar .log (0/1)")
    artefatos.add_argument("--save-metrics", type=int, default=1, help="Salvar .jsonl (0/1)")
    artefatos.add_argument("--save-summary", type=int, default=1, help="Salvar .json (0/1)")

    integracao = parser.add_argument_group("integração")
    integracao.add_argument("--sheer-cmd", default="")

    return parser


def parse_args() -> TrainArgs:
    parser = _build_parser()
    ns = parser.parse_args()

    run_id = ns.run_id or _default_run_id(ns.model, ns.foldsnet_dataset)

    return TrainArgs(
        backend="folds",
        epochs=ns.epochs,
        batch=ns.batch,
        lr=ns.lr,
        run_id=run_id,
        resume=ns.resume,
        init_checkpoint=ns.init_checkpoint,
        device=ns.device,
        console=ns.console,
        log_level=ns.log_level,
        log_file=ns.log_file,
        sheer_cmd=ns.sheer_cmd,
        model=ns.model,
        n_dendrites=ns.n_dendrites,
        n_synapses_per_dendrite=ns.n_synapses_per_dendrite,
        hidden=ns.hidden,
        threshold=ns.threshold,
        save_fold=ns.save_fold,
        save_mind=ns.save_mind,
        save_pt=ns.save_pt,
        save_log=ns.save_log,
        save_metrics=ns.save_metrics,
        save_summary=ns.save_summary,
        disable_stdp=ns.disable_stdp,
        disable_homeostase=ns.disable_homeostase,
        disable_inibicao=ns.disable_inibicao,
        disable_refratario=ns.disable_refratario,
        disable_backprop=ns.disable_backprop,
        disable_sfa=ns.disable_sfa,
        disable_stp=ns.disable_stp,
        disable_wave=ns.disable_wave,
        disable_circadian=ns.disable_circadian,
        disable_engram=ns.disable_engram,
        disable_speech=ns.disable_speech,
        foldsnet_variant=ns.foldsnet_variant,
        foldsnet_dataset=ns.foldsnet_dataset,
    )


if __name__ == "__main__":
    args = parse_args()
    _save_run_metadata(args)
    raise SystemExit(run_training(args))
# =============================================================================
# COMANDOS DE TREINO — COPIAR E COLAR NO POWERSHELL
# =============================================================================
#
# --- MNIST (50 épocas, salva fold + mind + pt + metrics + summary) ---
#
# python train_mnist_folds.py --model foldsnet --foldsnet-variant 4L --foldsnet-dataset mnist --epochs 50 --batch 64 --lr 1e-3 --device cpu --run-id foldsnet_4L_mnist_50ep --save-fold 1 --save-mind 1 --save-pt 1 --save-metrics 1 --save-summary 1 --console
#
# --- RETOMAR TREINO MNIST (resume a partir do checkpoint salvo) ---
#
# python train_mnist_folds.py --model foldsnet --foldsnet-variant 4L --foldsnet-dataset mnist --epochs 50 --batch 64 --lr 1e-3 --device cpu --run-id foldsnet_4L_mnist_50ep --save-fold 1 --save-mind 1 --save-pt 1 --save-metrics 1 --save-summary 1 --console --resume
#
# --- CIFAR10 (50 épocas, salva fold + mind + pt + metrics + summary) ---
#
# python train_mnist_folds.py --model foldsnet --foldsnet-variant 4L --foldsnet-dataset cifar10 --epochs 50 --batch 64 --lr 1e-3 --device cpu --run-id foldsnet_4L_cifar10_50ep --save-fold 1 --save-mind 1 --save-pt 1 --save-metrics 1 --save-summary 1 --console
#
# --- RETOMAR TREINO CIFAR10 ---
#
# python train_mnist_folds.py --model foldsnet --foldsnet-variant 4L --foldsnet-dataset cifar10 --epochs 50 --batch 64 --lr 1e-3 --device cpu --run-id foldsnet_4L_cifar10_50ep --save-fold 1 --save-mind 1 --save-pt 1 --save-metrics 1 --save-summary 1 --console --resume
#
# NOTA: O resume usa o checkpoint salvo automaticamente em runs/<run-id>/checkpoint.pt
# Para continuar de onde parou, use EXATAMENTE o mesmo --run-id do treino original.
# =============================================================================
