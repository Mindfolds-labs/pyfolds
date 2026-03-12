"""Serialização de modelos FOLDSNet em .fold e .mind."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import torch
from pyfolds.serialization.foldio import FoldReader, FoldWriter


def save_foldsnet(model: torch.nn.Module, path: str, fmt: str, metadata: dict[str, Any] | None = None) -> None:
    """Salva um modelo FOLDSNet em container fold/mind."""
    if fmt not in {"fold", "mind"}:
        raise ValueError("Formato inválido. Use apenas 'fold' ou 'mind'.")

    target = Path(path)
    if target.suffix != f".{fmt}":
        raise ValueError(f"Extensão inválida para formato {fmt}: esperado .{fmt}")

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    payload = {
        "class": model.__class__.__name__,
        "init_kwargs": model.get_init_kwargs(),
        "format": fmt,
        "metadata": metadata or {},
    }

    with FoldWriter(str(target), compress="zstd", zstd_level=9 if fmt == "fold" else 6) as writer:
        writer.add_chunk("torch_state", "TORC", buffer.getvalue())
        writer.add_chunk("model_manifest", "JSON", json.dumps(payload).encode("utf-8"))
        writer.finalize(metadata={"model_type": "FOLDSNet", "format": fmt})


def load_foldsnet(path: str, model_cls: type[torch.nn.Module], device: str = "cpu") -> torch.nn.Module:
    """Carrega um modelo FOLDSNet salvo em .fold/.mind."""
    with FoldReader(path, use_mmap=True) as reader:
        manifest = reader.read_json("model_manifest")
        state_bytes = reader.read_chunk_bytes("torch_state")

    model = model_cls(**manifest["init_kwargs"])
    state_dict = torch.load(io.BytesIO(state_bytes), map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
