# ADR Index — FOLD

## Lista de ADRs

- [ADR-001 — Governança de decisões para formato FOLD](ADR-001-governanca-decisoes-fold.md) *(Accepted)*
- [ADR-002 — Container `.fold/.mind` para checkpoints científicos](ADR-002-container-fold-mind.md) *(Accepted)*
- [ADR-003 — Integridade e ECC por chunk](ADR-003-integridade-ecc-por-chunk.md) *(Accepted)*
- [ADR-004 — Correções de regressão em refratário, serialização e API de camada](ADR-004-correcao-regressoes-refratario-serializacao-api.md) *(Accepted)*
- [ADR-006 — Limite anti-DoS para índice (`MAX_INDEX_SIZE`)](ADR-006-limite-max-index-size.md) *(Accepted)*
- [ADR-007 — Proposta de limite anti-DoS por chunk (`MAX_CHUNK_SIZE`)](ADR-007-proposta-max-chunk-size.md) *(Proposed)*
- [ADR-008 — Leitura parcial com índice + `mmap`](ADR-008-leitura-parcial-e-mmap.md) *(Accepted)*
- [ADR-009 — Segurança na desserialização Torch](ADR-009-seguranca-desserializacao-torch.md) *(Accepted)*

## Mapa de dependências

```text
ADR-001
  └─ ADR-002
      ├─ ADR-003
      │   └─ ADR-004
      ├─ ADR-006
      │   └─ ADR-007 (Proposed)
      └─ ADR-008
          └─ ADR-009

Dependência adicional:
ADR-009 também depende de ADR-003 (integridade antes de decode).
```

## Leitura recomendada

1. ADR-001
2. ADR-002
3. ADR-003
4. ADR-006
5. ADR-008
6. ADR-009
7. ADR-004
8. ADR-007 (proposta futura)
