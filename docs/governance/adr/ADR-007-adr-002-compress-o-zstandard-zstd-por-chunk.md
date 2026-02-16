# ADR-002 — Compressão Zstandard (ZSTD) por chunk

## Status
Accepted

## Contexto
Precisávamos reduzir tamanho de artefatos sem perder leitura parcial e sem alterar semântica dos chunks.

## Decisão
Usar compressão por chunk, com flag no header do chunk:
- `FLAG_COMP_NONE = 0`
- `FLAG_COMP_ZSTD = 1`

Padrão operacional: compressão `zstd` com nível moderado (`level=3`), com fallback explícito para `none` quando indisponível.

## Consequências
### Positivas
- melhor relação tamanho/latência para checkpoints reais;
- preserva granularidade por chunk.

### Trade-offs
- dependência opcional (`zstandard`);
- custo de CPU em gravação/leitura comprimida.

## Relacionamentos
- Depende do container definido em [ADR-001](./ADR-001-formato-binario-fold-mind.md).
- Interage com integridade/ECC definidos em [ADR-003](./ADR-003-ecc-opcional-por-chunk.md).
