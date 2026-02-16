# ADR Index — Serialização `.fold/.mind` e qualidade

## Índice
- [ADR-001 — Container único `.fold/.mind`](./ADR-001-container-fold-mind.md)
- [ADR-002 — Integridade por chunk com CRC32C/SHA256 e ECC opcional](./ADR-002-integridade-por-chunk-e-ecc.md)
- [ADR-003 — Desserialização segura de payload Torch](./ADR-003-desserializacao-segura-torch.md)
- [ADR-004 — Limites defensivos de I/O (`MAX_INDEX_SIZE`, `MAX_CHUNK_SIZE`)](./ADR-004-limites-defensivos-de-io.md)
- [ADR-005 — Leitura parcial orientada a índice com `mmap`](./ADR-005-leitura-parcial-com-indice-e-mmap.md)
- [ADR-006 — Manifesto, hashes hierárquicos e metadados de reprodutibilidade](./ADR-006-manifesto-e-reprodutibilidade.md)
- [ADR-007 — Plano de testes para robustez de serialização](./ADR-007-plano-de-testes-robustez-serializacao.md)
- [ADR-008 — Estabilidade de API e semântica refratária](./ADR-008-estabilidade-api-e-semantica-refrataria.md)
- [ADR-009 — Quality gates (lint/tipos/segurança/testes/benchmark)](./ADR-009-quality-gates-ci-benchmark.md)

## Dependências
- ADR-001 é base estrutural.
- ADR-002 depende de ADR-001.
- ADR-003 depende de ADR-001 e ADR-002.
- ADR-004 depende de ADR-001.
- ADR-005 depende de ADR-001 e ADR-004.
- ADR-006 depende de ADR-001 e ADR-002.
- ADR-007 depende de ADR-002, ADR-003 e ADR-004.
- ADR-008 é transversal (API/comportamento) e complementa ADR-007.
- ADR-009 integra e operacionaliza ADR-007 para CI/release.

## Fontes reaproveitadas
- `docs/developments/adr/ADR-0002-fold-mind-container.md`
- `docs/developments/adr/ADR-0003-test-failures-refractory-serialization-layer.md`
- `FOLD_QUALITY_PLAN.md`
