# ADR-003 — Desserialização segura de payload Torch

## Status
Accepted

## Contexto
`torch.load` em modo amplo pode carregar objetos arbitrários.

## Decisão
Carregar `torch_state` com `weights_only=True` por padrão e exigir modo trusted explícito quando necessário.

## Consequências
- + reduz superfície de risco em cenários não confiáveis.
- - pode exigir override em artefatos legados/confiáveis.
