# ADR-003 — ECC opcional por chunk

## Status
Accepted

## Contexto
Somente checksum detecta corrupção, mas não corrige erros em mídia/storage com falhas esparsas.

## Decisão
Adicionar ECC opcional por chunk, com perfil de proteção:
- `off` → `NoECC`
- `low` → `rs(16)`
- `med` → `rs(32)`
- `high` → `rs(64)`

Fluxo de escrita: compressão → checksums → ECC.
Fluxo de leitura: leitura → decode ECC → validação → descompressão.

## Consequências
### Positivas
- maior resiliência a corrupção localizada;
- granularidade fina de recuperação por chunk.

### Trade-offs
- overhead de CPU e bytes adicionais (`ecc_len`);
- dependência opcional para implementação Reed-Solomon.

## Relacionamentos
- Estruturado sobre [ADR-001](./ADR-001-formato-binario-fold-mind.md).
- Validado dentro da política de [ADR-004](./ADR-004-validacao-multicamada.md).
