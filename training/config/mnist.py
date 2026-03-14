from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ModelKind = Literal["mpjrd", "foldsnet"]
BackendKind = Literal["folds", "mind"]


@dataclass
class BaseTrainConfig:
    backend: BackendKind
    model: ModelKind
    epochs: int
    batch: int
    lr: float
    run_id: str
    resume: bool = False
    init_checkpoint: str = ""
    device: str = "cpu"
    console: bool = False
    log_level: str = "INFO"
    log_file: str = "train.log"
    sheer_cmd: str = ""
    timesteps: int = 4

    # output controls
    save_fold: int = 1
    save_mind: int = 1
    save_pt: int = 1
    save_log: int = 1
    save_metrics: int = 1
    save_summary: int = 1


@dataclass
class MPJRDTrainConfig:
    n_dendrites: int = 4
    n_synapses_per_dendrite: int = 32
    hidden: int = 128
    threshold: float = 0.45

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


@dataclass
class FOLDSNetTrainConfig:
    variant: str = "4L"
    dataset: str = "mnist"


@dataclass
class RunConfig:
    base: BaseTrainConfig
    mpjrd: MPJRDTrainConfig = field(default_factory=MPJRDTrainConfig)
    foldsnet: FOLDSNetTrainConfig = field(default_factory=FOLDSNetTrainConfig)

    def active_model_config(self) -> dict[str, Any]:
        if self.base.model == "mpjrd":
            return asdict(self.mpjrd)
        return asdict(self.foldsnet)


class ConfigValidationError(ValueError):
    """Raised when config compatibility checks fail."""


def validate_run_config(config: RunConfig) -> None:
    base = config.base
    if base.epochs <= 0:
        raise ConfigValidationError("epochs deve ser > 0")
    if base.batch <= 0:
        raise ConfigValidationError("batch deve ser > 0")
    if base.lr <= 0:
        raise ConfigValidationError("lr deve ser > 0")
    if base.model == "mpjrd":
        if config.mpjrd.n_dendrites <= 0:
            raise ConfigValidationError("n_dendrites deve ser > 0 para mpjrd")
        if config.mpjrd.n_synapses_per_dendrite <= 0:
            raise ConfigValidationError("n_synapses_per_dendrite deve ser > 0 para mpjrd")
        if config.mpjrd.hidden <= 0:
            raise ConfigValidationError("hidden deve ser > 0 para mpjrd")
    if base.model == "foldsnet":
        if not config.foldsnet.variant:
            raise ConfigValidationError("variant não pode ser vazio para foldsnet")
        if config.foldsnet.dataset not in {"mnist", "cifar10", "cifar100"}:
            raise ConfigValidationError("dataset inválido para foldsnet: use mnist, cifar10 ou cifar100")


def serialize_config(config: RunConfig) -> dict[str, Any]:
    return {
        "base": asdict(config.base),
        "mpjrd": asdict(config.mpjrd),
        "foldsnet": asdict(config.foldsnet),
        "active_model": config.active_model_config(),
    }
