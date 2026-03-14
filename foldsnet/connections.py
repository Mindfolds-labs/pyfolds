"""Utilitários para conexões esparsas da FOLDSNet."""

from __future__ import annotations

import torch


def _create_sparse_connections(n_lgn: int, n_v1: int, n_it: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cria máscaras de conexão LGN→V1 (30%) e V1→IT (15%)."""

    def _sparse_mask(n_out: int, n_in: int, rate: float) -> torch.Tensor:
        """Gera máscara esparsa (n_out × n_in) com taxa de conexão `rate`."""
        k = max(1, int(n_in * rate))
        # vetorizado via argsort; equivalente semântico ao loop anterior
        noise = torch.rand(n_out, n_in)
        indices = noise.argsort(dim=1)[:, :k]
        mask = torch.zeros(n_out, n_in)
        mask.scatter_(1, indices, 1.0)
        return mask

    lgn_to_v1 = _sparse_mask(n_v1, n_lgn, 0.30)
    v1_to_it = _sparse_mask(n_it, n_v1, 0.15)

    return lgn_to_v1, v1_to_it
