# ADR-006 — Manifesto, hashes hierárquicos e metadados de reprodutibilidade

## Status
Accepted

## Contexto
Checkpoints científicos exigem trilha de auditoria e rastreabilidade de ambiente.

## Decisão
Persistir manifesto com rota de chunks e metadados de reproducibilidade (Python/Torch/plataforma/git/seed), além de `chunk_hashes` e `manifest_hash`.

## Consequências
- + melhor governança de experimentos.
- - metadados extras aumentam tamanho do índice.
