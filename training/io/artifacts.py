from __future__ import annotations

from pathlib import Path
from typing import Any

from serialization.folds_io import save_model_fold
from serialization.mind_io import save_model_mind


def save_backend_artifact(backend: str, path: Path, payload: dict[str, Any]) -> None:
    if backend == "folds":
        save_model_fold(path, payload)
    else:
        save_model_mind(path, payload)
