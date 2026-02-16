# ADR-004 — Limites defensivos de I/O (`MAX_INDEX_SIZE`, `MAX_CHUNK_SIZE`)

## Status
Accepted

## Contexto
Leituras/escritas sem limites podem causar exaustão de memória e falhas difíceis de diagnosticar.

## Decisão
Definir e validar limites explícitos:
- `MAX_INDEX_SIZE = 100 MiB`
- `MAX_CHUNK_SIZE = 2 GiB`

Aplicar em leitura e escrita.

## Consequências
- + parser mais resiliente a arquivos malformados.
- - arquivos acima dos limites são rejeitados explicitamente.
