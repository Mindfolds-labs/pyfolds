from __future__ import annotations

from pathlib import Path
from typing import Any

from .folds_io import load_model_fold, save_model_fold


def save_model_mind(path: str | Path, payload: dict[str, Any]) -> None:
    save_model_fold(path, {**payload, "_mind_backend": True})


def load_model_mind(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return load_model_fold(path, map_location=map_location)
