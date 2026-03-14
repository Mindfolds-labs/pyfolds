from __future__ import annotations

import argparse
import json
import secrets
import string
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


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Treino MNIST com backend folds")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", default="", help="Checkpoint .pt para warm start (carrega pesos)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--console", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default="train.log")
    parser.add_argument("--sheer-cmd", default="")
    parser.add_argument("--model", choices=["mpjrd", "foldsnet"], default="foldsnet")

    # Parâmetros de arquitetura MPJRD
    parser.add_argument("--n-dendrites", type=int, default=4, help="Número de dendritos")
    parser.add_argument("--n-synapses", type=int, default=32, dest="n_synapses_per_dendrite", help="Sinapses por dendrito")
    parser.add_argument("--hidden", type=int, default=128, help="Número de neurônios excitatórios")
    parser.add_argument("--foldsnet-variant", default="4L")
    parser.add_argument("--foldsnet-dataset", default="mnist")
    parser.add_argument("--threshold", type=float, default=0.45, help="Limiar inicial de disparo")

    # Controle de formatos de saída
    parser.add_argument("--save-fold", type=int, default=1, help="Salvar .fold (0/1)")
    parser.add_argument("--save-mind", type=int, default=1, help="Salvar .mind (0/1)")
    parser.add_argument("--save-pt", type=int, default=1, help="Salvar .pt (0/1)")
    parser.add_argument("--save-log", type=int, default=1, help="Salvar .log (0/1)")
    parser.add_argument("--save-metrics", type=int, default=1, help="Salvar .jsonl (0/1)")
    parser.add_argument("--save-summary", type=int, default=1, help="Salvar .json (0/1)")

    # Controle de mecanismos (desabilitar)
    parser.add_argument("--disable-stdp", action="store_true", help="Desativa STDP")
    parser.add_argument("--disable-homeostase", action="store_true", help="Desativa Homeostase")
    parser.add_argument("--disable-inibicao", action="store_true", help="Desativa Inibição")
    parser.add_argument("--disable-refratario", action="store_true", help="Desativa Período refratário")
    parser.add_argument("--disable-backprop", action="store_true", help="Desativa Backpropagação")
    parser.add_argument("--disable-sfa", action="store_true", help="Desativa Adaptação SFA")
    parser.add_argument("--disable-stp", action="store_true", help="Desativa Dinâmica STP")
    parser.add_argument("--disable-wave", action="store_true", help="Desativa mecanismo Wave")
    parser.add_argument("--disable-circadian", action="store_true", help="Desativa mecanismo Circadiano")
    parser.add_argument("--disable-engram", action="store_true", help="Desativa memória por Engrams")
    parser.add_argument("--disable-speech", action="store_true", help="Desativa Speech tracking")

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
