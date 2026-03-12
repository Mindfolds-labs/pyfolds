"""Conexões esparsas entre camadas da FOLDSNet."""

from __future__ import annotations

import torch


def _create_sparse_connections(
    n_lgn: int,
    n_v1: int,
    n_it: int,
    lgn_to_v1_ratio: float = 0.30,
    v1_to_it_ratio: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cria máscaras esparsas LGN→V1 e V1→IT."""
    lgn_to_v1 = torch.zeros(n_v1, n_lgn)
    for i in range(n_v1):
        n_conn = max(1, int(n_lgn * lgn_to_v1_ratio))
        indices = torch.randperm(n_lgn)[:n_conn]
        lgn_to_v1[i, indices] = 1.0

    v1_to_it = torch.zeros(n_it, n_v1)
    for i in range(n_it):
        n_conn = max(1, int(n_v1 * v1_to_it_ratio))
        indices = torch.randperm(n_v1)[:n_conn]
        v1_to_it[i, indices] = 1.0

    return lgn_to_v1, v1_to_it
