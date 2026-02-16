# ADR-008 — Leitura parcial com índice + `mmap`

## Status
Accepted

## Contexto
Fluxos científicos precisam abrir rapidamente metadados/chunks específicos sem materializar todo o checkpoint.

## Decisão
Usar índice com offsets explícitos e suporte a `mmap` no reader para leitura por faixa (`offset`, `length`) com bounds-check.

## Consequências
### Positivas
- menor custo de I/O em inspeções parciais;
- melhor escalabilidade para artefatos grandes.

### Trade-offs
- maior complexidade de validação de limites e tratamento de EOF.

## Dependências
- ADR-002
- ADR-006
