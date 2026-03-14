from __future__ import annotations

import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import torch

from training.config.mnist import RunConfig
from training.models.contracts import ModelMetadata


def setup_logger(run_dir: Path, log_file: str, console: bool) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"mnist.{run_dir.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(run_dir / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def _format_line(content: str, width: int, align: str = "left") -> str:
    padded = content.ljust(width) if align == "left" else content.center(width)
    return f"║ {padded} ║"




def _fit_line(content: str, width: int) -> str:
    if len(content) <= width:
        return content
    if width <= 3:
        return content[:width]
    return content[: width - 3] + "..."

def _print_box(title: str, lines: list[str], width: int) -> None:
    print(f"╔{'═' * (width + 2)}╗")
    if title:
        print(_format_line(title, width, "center"))
        print(f"╠{'═' * (width + 2)}╣")
    for line in lines:
        print(_format_line(line, width, "left"))
    print(f"╚{'═' * (width + 2)}╝")


def print_layout(config: RunConfig, metadata: ModelMetadata, mpjrd_cfg: object | None) -> None:
    if not config.base.console:
        return

    import shutil

    box_width = min(78, shutil.get_terminal_size().columns - 4)
    header = [
        _fit_line(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Run ID: {config.base.run_id}", box_width),
        _fit_line(f"Framework: PyFolds | Backend: PyTorch {torch.__version__}", box_width),
        _fit_line(f"Dispositivo: {config.base.device.upper()} | Modelo: {metadata.family.upper()}", box_width),
    ]
    _print_box("", header, box_width)
    print()

    model_lines = [
        f"Modelo base        : {metadata.family.upper()}",
        f"Epochs             : {config.base.epochs}",
        f"Learning rate      : {config.base.lr}",
        f"Batch size         : {config.base.batch}",
    ]
    if metadata.family == "mpjrd" and mpjrd_cfg is not None:
        model_lines.extend(
            [
                f"Dendritos          : {mpjrd_cfg.n_dendrites}",
                f"Sinapses/dendrito  : {mpjrd_cfg.n_synapses_per_dendrite}",
                f"Hidden dim         : {config.mpjrd.hidden}",
                f"Threshold inicial  : {mpjrd_cfg.theta_init}",
            ]
        )
        total_synapses = mpjrd_cfg.n_dendrites * mpjrd_cfg.n_synapses_per_dendrite
        vc_approx = total_synapses * math.log2(total_synapses + 1)
        model_lines.append(f"VC-dimension aprox : {vc_approx:.1e}")
    else:
        model_lines.extend(
            [
                f"Variant            : {config.foldsnet.variant}",
                f"Dataset            : {config.foldsnet.dataset}",
                "Mecanismos MPJRD   : ativos (plasticidade, homeostase, inibição,",
                "                      adaptação, backprop, wave, circadiano, engram)",
            ]
        )

    _print_box("CONFIGURAÇÃO", model_lines, box_width)
