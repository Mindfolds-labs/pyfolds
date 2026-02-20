"""Exemplo de execução silenciosa com logging em arquivo.

- Progresso via print() no terminal.
- Logs completos (incluindo stacktrace) no arquivo incremental.
- Exit code != 0 em erro para integração com PowerShell/CI.
"""

from __future__ import annotations

import sys
from pathlib import Path
import logging

import torch

import pyfolds
from pyfolds.utils.logging import setup_run_logging


def main(logger: logging.Logger, log_path: Path) -> int:
    print("▶ Iniciando execução MNIST silent...")
    logger.info("START run_mnist_silent log_path=%s", log_path)

    # Mini fluxo de demonstração (sem depender de download de dataset)
    cfg = pyfolds.NeuronConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    layer = pyfolds.AdaptiveNeuronLayer(n_neurons=3, cfg=cfg)

    print("▶ Rodando inferência sintética...")
    x = torch.rand(8, 3, 2, 4)
    out = layer(x, mode="inference")

    logger.info("Output spikes shape=%s", tuple(out["spikes"].shape))
    print(f"✅ Finalizado (log salvo em {log_path})")
    return 0


if __name__ == "__main__":
    logger, log_path = setup_run_logging(
        app="pyfolds",
        version=getattr(pyfolds, "__version__", "unknown"),
        log_dir="logs",
        level="INFO",
        console=False,
        fixed_layout=True,
        structured=False,
    )

    try:
        code = main(logger, log_path)
    except Exception as exc:  # pragma: no cover (script example)
        logger.exception("FALHA FATAL")
        print(f"❌ Erro: {exc} (ver log: {log_path})")
        code = 1
    sys.exit(code)
