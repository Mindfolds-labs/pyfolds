"""Wrappers para serialização .fold e .mind da FOLDSNet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


_VALID_FORMATS = {"fold", "mind"}


def _normalize_format(fmt: str) -> str:
    normalized = (fmt or "").strip().lower()
    if normalized not in _VALID_FORMATS:
        raise ValueError(f"Formato inválido '{fmt}'. Use: {sorted(_VALID_FORMATS)}")
    return normalized


def _validate_extension(path: str, fmt: str) -> None:
    expected = f".{fmt}"
    if Path(path).suffix.lower() != expected:
        raise ValueError(f"Extensão incompatível: esperado arquivo '{expected}' para fmt='{fmt}', recebido '{path}'.")


def save_payload(path: str, fmt: str, payload: dict[str, Any]) -> None:
    """Salva payload em formato .fold ou .mind."""
    normalized = _normalize_format(fmt)
    _validate_extension(path, normalized)
    torch.save(payload, path)


def load_payload(path: str, fmt: str, map_location: str = "cpu") -> dict[str, Any]:
    """Carrega payload em formato .fold ou .mind."""
    normalized = _normalize_format(fmt)
    _validate_extension(path, normalized)
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado para carga FOLDSNet: {path}")

    return torch.load(
        in_path,
        map_location=map_location,
        weights_only=False,
    )
