from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pyfolds.serialization.foldio import FoldReader, FoldWriter


def save_model_fold(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(payload, buf)
    with FoldWriter(str(target), compress="zstd") as writer:
        writer.add_chunk("torch_state", "TSAV", buf.getvalue())
        writer.add_chunk("llm_manifest", "JSON", json.dumps({"backend": "folds"}).encode("utf-8"))
        writer.finalize(metadata={"format": "fold"})


def load_model_fold(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    with FoldReader(str(path), use_mmap=True) as reader:
        return reader.read_torch("torch_state", map_location=map_location)
