from __future__ import annotations

from dataclasses import dataclass

from training.config.mnist import BaseTrainConfig, FOLDSNetTrainConfig, MPJRDTrainConfig, RunConfig
from training.trainers.mnist_trainer import run_mnist_training


@dataclass
class TrainArgs:
    """Legacy args mantidos para compatibilidade dos scripts existentes."""

    backend: str
    epochs: int
    batch: int
    lr: float
    run_id: str
    resume: bool
    device: str
    console: bool
    log_level: str
    log_file: str
    sheer_cmd: str = ""
    model: str = "mpjrd"
    timesteps: int = 4

    n_dendrites: int = 4
    n_synapses_per_dendrite: int = 32
    hidden: int = 128
    threshold: float = 0.45

    save_fold: int = 1
    save_mind: int = 1
    save_pt: int = 1
    save_log: int = 1
    save_metrics: int = 1
    save_summary: int = 1

    disable_stdp: bool = False
    disable_homeostase: bool = False
    disable_inibicao: bool = False
    disable_refratario: bool = False
    disable_backprop: bool = False
    disable_sfa: bool = False
    disable_stp: bool = False
    disable_wave: bool = False
    disable_circadian: bool = False
    disable_engram: bool = False
    disable_speech: bool = False

    foldsnet_variant: str = "4L"
    foldsnet_dataset: str = "mnist"


def _to_run_config(args: TrainArgs) -> RunConfig:
    base = BaseTrainConfig(
        backend=args.backend,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        run_id=args.run_id,
        resume=args.resume,
        device=args.device,
        console=args.console,
        log_level=args.log_level,
        log_file=args.log_file,
        sheer_cmd=args.sheer_cmd,
        timesteps=args.timesteps,
        save_fold=args.save_fold,
        save_mind=args.save_mind,
        save_pt=args.save_pt,
        save_log=args.save_log,
        save_metrics=args.save_metrics,
        save_summary=args.save_summary,
    )
    mpjrd = MPJRDTrainConfig(
        n_dendrites=args.n_dendrites,
        n_synapses_per_dendrite=args.n_synapses_per_dendrite,
        hidden=args.hidden,
        threshold=args.threshold,
        disable_stdp=args.disable_stdp,
        disable_homeostase=args.disable_homeostase,
        disable_inibicao=args.disable_inibicao,
        disable_refratario=args.disable_refratario,
        disable_backprop=args.disable_backprop,
        disable_sfa=args.disable_sfa,
        disable_stp=args.disable_stp,
        disable_wave=args.disable_wave,
        disable_circadian=args.disable_circadian,
        disable_engram=args.disable_engram,
        disable_speech=args.disable_speech,
    )
    foldsnet = FOLDSNetTrainConfig(variant=args.foldsnet_variant, dataset=args.foldsnet_dataset)
    return RunConfig(base=base, mpjrd=mpjrd, foldsnet=foldsnet)


def run_training(args: TrainArgs) -> int:
    return run_mnist_training(_to_run_config(args))
