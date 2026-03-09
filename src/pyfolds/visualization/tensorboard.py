"""TensorBoard observability helpers for PyFolds training and Noetic dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence


class TensorBoardLogger:
    """Thin wrapper around torch SummaryWriter with optional dependency checks."""

    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            raise ImportError(
                "TensorBoard logger requer `torch` + `tensorboard`. "
                "Instale com `pip install tensorboard`."
            ) from exc

        self.log_dir = str(Path(log_dir).expanduser().resolve())
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=self.log_dir)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self._writer.add_scalar(name, value, step)

    def log_histogram(self, name: str, values, step: int) -> None:
        self._writer.add_histogram(name, values, step)

    def log_embedding(self, features, *, metadata: Optional[Sequence[str]] = None, step: int = 0, tag: str = "emb") -> None:
        self._writer.add_embedding(features, metadata=metadata, global_step=step, tag=tag)

    def log_engram_metrics(self, *, step: int, wave_activity: float, engram_count: int, consolidation: float, pruning: float) -> None:
        self.log_scalar("noetic/wave_activity", float(wave_activity), step)
        self.log_scalar("noetic/engrams", float(engram_count), step)
        self.log_scalar("noetic/consolidation", float(consolidation), step)
        self.log_scalar("noetic/pruning", float(pruning), step)

    def log_specialization_distribution(self, distribution: Mapping[str, float], step: int) -> None:
        for area, value in distribution.items():
            self.log_scalar(f"specialization/{area}", float(value), step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
