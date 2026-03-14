"""Wrappers para serialização .fold e .mind da FOLDSNet."""

from __future__ import annotations

from typing import Any

import torch


def save_payload(path: str, fmt: str, payload: dict[str, Any]) -> None:
    """Salva payload em formato .fold ou .mind."""
    if fmt not in {"fold", "mind"}:
        raise ValueError("Formato inválido. Use 'fold' ou 'mind'.")
    torch.save(payload, path)


def load_payload(path: str, fmt: str, map_location: str = "cpu") -> dict[str, Any]:
    """Carrega payload em formato .fold ou .mind."""
    if fmt not in {"fold", "mind"}:
        raise ValueError("Formato inválido. Use 'fold' ou 'mind'.")
    # weights_only=False necessário: payload contém dicts e
    # escalares além de tensores. Revisar ao migrar para formato
    # puramente baseado em tensores.
    return torch.load(path, map_location=map_location, weights_only=False)
