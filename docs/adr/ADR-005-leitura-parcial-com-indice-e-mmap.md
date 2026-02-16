# ADR-005 — Leitura parcial orientada a índice com `mmap`

## Status
Accepted

## Contexto
Fluxos de auditoria não precisam carregar `state_dict` completo.

## Decisão
Permitir inspeção de header/índice e leitura de chunks individuais com suporte a `mmap`.

## Consequências
- + menor latência para diagnósticos e analytics.
- - necessidade de validações rígidas de offset/EOF.
