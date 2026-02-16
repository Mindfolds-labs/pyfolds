# ADR-004 — Validação multicamada para leitura segura

## Status
Accepted

## Contexto
O parser precisa ser robusto contra truncamento, offsets inválidos, corrupção de payload e risco de desserialização insegura.

## Decisão
Padronizar validação por camadas:
1. Estrutural (header, magic, offsets);
2. Limites anti-DoS (`MAX_INDEX_SIZE`, `MAX_CHUNK_SIZE`);
3. Bounds/EOF em toda leitura (`offset + length <= file_size`);
4. Integridade por chunk (CRC32C + SHA-256 + hash hierárquico);
5. Manifesto (`manifest_hash`);
6. Desserialização segura (ex.: `torch.load(weights_only=True)`).

## Consequências
### Positivas
- redução de superfície de falha/ataque;
- mensagens de erro acionáveis por camada.

### Trade-offs
- aumento de complexidade no fluxo de leitura;
- pequenas penalidades de desempenho com verificações completas.

## Relacionamentos
- Consolida o modelo de [ADR-001](./ADR-001-formato-binario-fold-mind.md), [ADR-002](./ADR-002-compressao-zstd-por-chunk.md) e [ADR-003](./ADR-003-ecc-opcional-por-chunk.md).
