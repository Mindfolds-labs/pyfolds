# ADR-006 — Limite anti-DoS para índice (`MAX_INDEX_SIZE`)

## Status
Accepted

## Contexto
O índice JSON fica no fim do arquivo e é carregado na abertura. Sem limite, um arquivo malicioso pode forçar alocação excessiva.

## Decisão
Manter limite rígido `MAX_INDEX_SIZE = 100 MiB` para `index_len` durante leitura.

## Consequências
### Positivas
- fail-fast em arquivos hostis/corrompidos;
- redução do risco de OOM e latência extrema.

### Trade-offs
- arquivos com índice acima do limite são rejeitados.

## Dependências
- ADR-002.
