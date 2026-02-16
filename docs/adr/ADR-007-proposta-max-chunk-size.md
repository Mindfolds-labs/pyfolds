# ADR-007 — Proposta de limite anti-DoS por chunk (`MAX_CHUNK_SIZE`)

## Status
Proposed

## Contexto
Mesmo com índice limitado, um chunk pode declarar `comp_len`/`uncomp_len` gigantes e induzir consumo excessivo de memória/CPU.

## Decisão
Propor limite explícito por chunk:
- `MAX_CHUNK_SIZE` aplicado a `comp_len` e `uncomp_len`.
- Valor inicial recomendado: **1 GiB por chunk** (com possibilidade de override avançado).

## Consequências esperadas
### Positivas
- proteção adicional contra ataques de exaustão de recursos;
- validação antecipada antes de leitura/descompressão custosa.

### Trade-offs
- chunks legítimos acima do limite exigem ajuste de configuração.

## Dependências
- ADR-006.
