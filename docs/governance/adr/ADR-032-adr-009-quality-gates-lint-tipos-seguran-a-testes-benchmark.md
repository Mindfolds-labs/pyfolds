# ADR-009 — Quality gates (lint/tipos/segurança/testes/benchmark)

## Status
Accepted

## Contexto
O plano de qualidade (`A1..A4`) exige padronização de validação antes de release.

## Decisão
Formalizar gates de qualidade em CI e rotina local:
- lint/tipos/segurança (`ruff`, `mypy`, `bandit`)
- testes unitários/integrados de serialização
- benchmarks com artefato versionado

## Consequências
- + maior confiança em releases.
- - pipelines mais longos.
