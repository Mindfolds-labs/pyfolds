"""Wrappers para serialização .fold e .mind da FOLDSNet."""

from __future__ import annotations

from typing import Any

import torch


_VALID_FORMATS = {"fold", "mind"}


def save_payload(path: str, fmt: str, payload: dict[str, Any]) -> None:
    """Salva payload em formato .fold ou .mind.

    Args:
        path: Caminho do arquivo de saída.
        fmt: Formato ('fold' ou 'mind').
        payload: Dicionário com state_dict e metadados.

    Raises:
        ValueError: Se o formato for inválido.
    """
    if fmt not in _VALID_FORMATS:
        raise ValueError(f"Formato inválido '{fmt}'. Use: {sorted(_VALID_FORMATS)}")
    torch.save(payload, path)


def load_payload(path: str, fmt: str, map_location: str = "cpu") -> dict[str, Any]:
    """Carrega payload em formato .fold ou .mind.

    Args:
        path: Caminho do arquivo.
        fmt: Formato ('fold' ou 'mind').
        map_location: Device de destino ('cpu' ou 'cuda').

    Raises:
        ValueError: Se o formato for inválido.
    """
    if fmt not in _VALID_FORMATS:
        raise ValueError(f"Formato inválido '{fmt}'. Use: {sorted(_VALID_FORMATS)}")
    # weights_only=False necessário pois payload contém dicts arbitrários,
    # não apenas tensores — manter False enquanto o formato incluir metadados.
    return torch.load(path, map_location=map_location, weights_only=False)
