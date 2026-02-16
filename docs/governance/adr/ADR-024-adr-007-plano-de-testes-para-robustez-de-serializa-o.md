# ADR-007 — Plano de testes para robustez de serialização

## Status
Accepted

## Contexto
Hardening de I/O requer validação contínua em cenários de corrupção e limites.

## Decisão
Manter suíte com testes de:
- roundtrip `.fold/.mind`
- corrupção por bitflip
- validação de magic/header/index
- vetor conhecido de CRC32C
- limites defensivos de chunk/index

## Consequências
- + regressões detectadas cedo.
- - custo adicional de manutenção de testes.
