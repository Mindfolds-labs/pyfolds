# Índice de Registros de Decisão Arquitetural (ADR)

## Árvore oficial

```text
/docs/governance/adr/
├── ADR-*.md
├── INDEX.md
├── README.md
└── legado/
    ├── 0001-import-contract-and-release-readiness.md
    ├── 0040-conclusao-do-ciclo-issue-e-foco-em-execucao.md
    └── ADR-038-governanca-fold-mind-auditoria-integridade-e-prompt-operacional.md
```

## ADRs canônicos (fonte oficial)

| ID | Título | Status | Data | Arquivo |
| :--- | :--- | :--- | :--- | :--- |
| ADR-001 | ADR-0002 — Container `.fold/.mind` para checkpoints científicos | Ativo | 2026-02-16 | [ADR-001-adr-0002-container-fold-mind-para-checkpoints-cient-ficos.md](ADR-001-adr-0002-container-fold-mind-para-checkpoints-cient-ficos.md) |
| ADR-002 | ADR-0003 — Correções de regressão em refratário, serialização e API de camada | Ativo | 2026-02-16 | [ADR-002-adr-0003-corre-es-de-regress-o-em-refrat-rio-serializa-o-e-api-de-camada.md](ADR-002-adr-0003-corre-es-de-regress-o-em-refrat-rio-serializa-o-e-api-de-camada.md) |
| ADR-003 | ADR-001 — Container único `.fold/.mind` | Ativo | 2026-02-16 | [ADR-003-adr-001-container-nico-fold-mind.md](ADR-003-adr-001-container-nico-fold-mind.md) |
| ADR-004 | ADR-001 — Formalização da especificação `.fold/.mind` | Ativo | 2026-02-16 | [ADR-004-adr-001-formaliza-o-da-especifica-o-fold-mind.md](ADR-004-adr-001-formaliza-o-da-especifica-o-fold-mind.md) |
| ADR-005 | ADR-001 — Formato binário `.fold/.mind` com índice no trailer | Ativo | 2026-02-16 | [ADR-005-adr-001-formato-bin-rio-fold-mind-com-ndice-no-trailer.md](ADR-005-adr-001-formato-bin-rio-fold-mind-com-ndice-no-trailer.md) |
| ADR-006 | ADR-001 — Governança de decisões para formato FOLD | Ativo | 2026-02-16 | [ADR-006-adr-001-governan-a-de-decis-es-para-formato-fold.md](ADR-006-adr-001-governan-a-de-decis-es-para-formato-fold.md) |
| ADR-007 | ADR-002 — Compressão Zstandard (ZSTD) por chunk | Ativo | 2026-02-16 | [ADR-007-adr-002-compress-o-zstandard-zstd-por-chunk.md](ADR-007-adr-002-compress-o-zstandard-zstd-por-chunk.md) |
| ADR-008 | ADR-002 — Container `.fold/.mind` para checkpoints científicos | Ativo | 2026-02-16 | [ADR-008-adr-002-container-fold-mind-para-checkpoints-cient-ficos.md](ADR-008-adr-002-container-fold-mind-para-checkpoints-cient-ficos.md) |
| ADR-009 | ADR-002 — Integridade por chunk com CRC32C/SHA256 e ECC opcional | Ativo | 2026-02-16 | [ADR-009-adr-002-integridade-por-chunk-com-crc32c-sha256-e-ecc-opcional.md](ADR-009-adr-002-integridade-por-chunk-com-crc32c-sha256-e-ecc-opcional.md) |
| ADR-010 | ADR-003 — Correção de regressões e hardening inicial | Ativo | 2026-02-16 | [ADR-010-adr-003-corre-o-de-regress-es-e-hardening-inicial.md](ADR-010-adr-003-corre-o-de-regress-es-e-hardening-inicial.md) |
| ADR-011 | ADR-003 — Desserialização segura de payload Torch | Ativo | 2026-02-16 | [ADR-011-adr-003-desserializa-o-segura-de-payload-torch.md](ADR-011-adr-003-desserializa-o-segura-de-payload-torch.md) |
| ADR-012 | ADR-003 — ECC opcional por chunk | Ativo | 2026-02-16 | [ADR-012-adr-003-ecc-opcional-por-chunk.md](ADR-012-adr-003-ecc-opcional-por-chunk.md) |
| ADR-013 | ADR-003 — Integridade e ECC por chunk | Ativo | 2026-02-16 | [ADR-013-adr-003-integridade-e-ecc-por-chunk.md](ADR-013-adr-003-integridade-e-ecc-por-chunk.md) |
| ADR-014 | ADR-004 — Correções de regressão em refratário, serialização e API de camada | Ativo | 2026-02-16 | [ADR-014-adr-004-corre-es-de-regress-o-em-refrat-rio-serializa-o-e-api-de-camada.md](ADR-014-adr-004-corre-es-de-regress-o-em-refrat-rio-serializa-o-e-api-de-camada.md) |
| ADR-015 | ADR-004 — Limites defensivos de I/O (`MAX_INDEX_SIZE`, `MAX_CHUNK_SIZE`) | Ativo | 2026-02-16 | [ADR-015-adr-004-limites-defensivos-de-i-o-max-index-size-max-chunk-size.md](ADR-015-adr-004-limites-defensivos-de-i-o-max-index-size-max-chunk-size.md) |
| ADR-016 | ADR-004 — Política de versionamento e compatibilidade do formato | Ativo | 2026-02-16 | [ADR-016-adr-004-pol-tica-de-versionamento-e-compatibilidade-do-formato.md](ADR-016-adr-004-pol-tica-de-versionamento-e-compatibilidade-do-formato.md) |
| ADR-017 | ADR-004 — Validação multicamada para leitura segura | Ativo | 2026-02-16 | [ADR-017-adr-004-valida-o-multicamada-para-leitura-segura.md](ADR-017-adr-004-valida-o-multicamada-para-leitura-segura.md) |
| ADR-018 | ADR-005 — Leitura parcial orientada a índice com `mmap` | Ativo | 2026-02-16 | [ADR-018-adr-005-leitura-parcial-orientada-a-ndice-com-mmap.md](ADR-018-adr-005-leitura-parcial-orientada-a-ndice-com-mmap.md) |
| ADR-019 | ADR-006 — Invariantes e estabilidade numérica como gate de qualidade | Ativo | 2026-02-16 | [ADR-019-adr-006-invariantes-e-estabilidade-num-rica-como-gate-de-qualidade.md](ADR-019-adr-006-invariantes-e-estabilidade-num-rica-como-gate-de-qualidade.md) |
| ADR-020 | ADR-006 — Limite anti-DoS para índice (`MAX_INDEX_SIZE`) | Ativo | 2026-02-16 | [ADR-020-adr-006-limite-anti-dos-para-ndice-max-index-size.md](ADR-020-adr-006-limite-anti-dos-para-ndice-max-index-size.md) |
| ADR-021 | ADR-006 — Manifesto, hashes hierárquicos e metadados de reprodutibilidade | Ativo | 2026-02-16 | [ADR-021-adr-006-manifesto-hashes-hier-rquicos-e-metadados-de-reprodutibilidade.md](ADR-021-adr-006-manifesto-hashes-hier-rquicos-e-metadados-de-reprodutibilidade.md) |
| ADR-022 | ADR-006 — Safe Weight Law (clamp + validação numérica) | Ativo | 2026-02-16 | [ADR-022-adr-006-safe-weight-law-clamp-valida-o-num-rica.md](ADR-022-adr-006-safe-weight-law-clamp-valida-o-num-rica.md) |
| ADR-023 | ADR-007 — Monitoramento periódico de invariantes (`N`, `I`, `theta`) | Ativo | 2026-02-16 | [ADR-023-adr-007-monitoramento-peri-dico-de-invariantes-n-i-theta.md](ADR-023-adr-007-monitoramento-peri-dico-de-invariantes-n-i-theta.md) |
| ADR-024 | ADR-007 — Plano de testes para robustez de serialização | Ativo | 2026-02-16 | [ADR-024-adr-007-plano-de-testes-para-robustez-de-serializa-o.md](ADR-024-adr-007-plano-de-testes-para-robustez-de-serializa-o.md) |
| ADR-025 | ADR-007 — Proposta de limite anti-DoS por chunk (`MAX_CHUNK_SIZE`) | Ativo | 2026-02-16 | [ADR-025-adr-007-proposta-de-limite-anti-dos-por-chunk-max-chunk-size.md](ADR-025-adr-007-proposta-de-limite-anti-dos-por-chunk-max-chunk-size.md) |
| ADR-026 | ADR-007 — Testes de corrupção e recuperação segura | Ativo | 2026-02-16 | [ADR-026-adr-007-testes-de-corrup-o-e-recupera-o-segura.md](ADR-026-adr-007-testes-de-corrup-o-e-recupera-o-segura.md) |
| ADR-027 | ADR-008 — Benchmark contínuo de serialização e escala | Ativo | 2026-02-16 | [ADR-027-adr-008-benchmark-cont-nuo-de-serializa-o-e-escala.md](ADR-027-adr-008-benchmark-cont-nuo-de-serializa-o-e-escala.md) |
| ADR-028 | ADR-008 — Estabilidade de API e semântica refratária | Ativo | 2026-02-16 | [ADR-028-adr-008-estabilidade-de-api-e-sem-ntica-refrat-ria.md](ADR-028-adr-008-estabilidade-de-api-e-sem-ntica-refrat-ria.md) |
| ADR-029 | ADR-008 — Controle homeostático com estratégia anti-windup | Ativo | 2026-02-16 | [ADR-029-adr-008-controle-homeost-tico-com-estrat-gia-anti-windup.md](ADR-029-adr-008-controle-homeost-tico-com-estrat-gia-anti-windup.md) |
| ADR-030 | ADR-008 — Leitura parcial com índice + `mmap` | Ativo | 2026-02-16 | [ADR-030-adr-008-leitura-parcial-com-ndice-mmap.md](ADR-030-adr-008-leitura-parcial-com-ndice-mmap.md) |
| ADR-031 | ADR-009 — Governança operacional e checklist de release | Ativo | 2026-02-16 | [ADR-031-adr-009-governan-a-operacional-e-checklist-de-release.md](ADR-031-adr-009-governan-a-operacional-e-checklist-de-release.md) |
| ADR-032 | ADR-009 — Quality gates (lint/tipos/segurança/testes/benchmark) | Ativo | 2026-02-16 | [ADR-032-adr-009-quality-gates-lint-tipos-seguran-a-testes-benchmark.md](ADR-032-adr-009-quality-gates-lint-tipos-seguran-a-testes-benchmark.md) |
| ADR-033 | ADR-009 — Segurança na desserialização Torch | Ativo | 2026-02-16 | [ADR-033-adr-009-seguran-a-na-desserializa-o-torch.md](ADR-033-adr-009-seguran-a-na-desserializa-o-torch.md) |
| ADR-034 | ADR-009 — Testes de propriedades matemáticas (property-based) | Ativo | 2026-02-16 | [ADR-034-adr-009-testes-de-propriedades-matem-ticas-property-based.md](ADR-034-adr-009-testes-de-propriedades-matem-ticas-property-based.md) |
| ADR-035 | Auditoria de `src/`, testes e estratégia de correção incremental | Ativo | 2026-02-17 | [ADR-035-auditoria-src-testes.md](ADR-035-auditoria-src-testes.md) |
| ADR-036 | Governança de validação integral pós-correção (import + testes + issue) | Ativo | 2026-02-17 | [ADR-036-governanca-validacao-integral-import-testes.md](ADR-036-governanca-validacao-integral-import-testes.md) |
| ADR-037 | Análise integral de execução (ISSUE-025) e atualização contínua de benchmark | Ativo | 2026-02-17 | [ADR-037-analise-integral-issue-025-benchmark-refresh.md](ADR-037-analise-integral-issue-025-benchmark-refresh.md) |
| ADR-038 | Auditoria contínua do formato `.fold/.mind` com governança de execução (ISSUE-036) | Ativo | 2026-02-18 | [ADR-038-auditoria-fold-mind-governanca-execucao.md](ADR-038-auditoria-fold-mind-governanca-execucao.md) |
| ADR-039 | Auditoria de prontidão para publicação no PyPI (ISSUE-038) | Ativo | 2026-02-18 | [ADR-039-auditoria-prontidao-publicacao-pypi.md](ADR-039-auditoria-prontidao-publicacao-pypi.md) |
| ADR-040 | Governança de execução: issues no HUB e plano ordenado 1→4 | Ativo | 2026-02-19 | [ADR-040-governanca-fluxo-issues-no-hub-e-planos-1-a-4.md](ADR-040-governanca-fluxo-issues-no-hub-e-planos-1-a-4.md) |
| ADR-041 | Ordem de execução de mixins, homeostase pós-refratário e contrato de saída de layer | Ativo | 2026-02-20 | [ADR-041-ordem-de-execucao-de-mixins-homeostase-pos-refratario-e-contrato-de-saida-de-layer.md](ADR-041-ordem-de-execucao-de-mixins-homeostase-pos-refratario-e-contrato-de-saida-de-layer.md) |
| ADR-042 | Governança de execução integral de testes e dossiê de qualidade | Ativo | 2026-02-20 | [ADR-042-governanca-de-execucao-integral-de-testes-e-dossie-de-qualidade.md](ADR-042-governanca-de-execucao-integral-de-testes-e-dossie-de-qualidade.md) |
| ADR-043 | Auditoria final integral de testes, imports e mecanismos | Ativo | 2026-02-20 | [ADR-043-auditoria-final-integral-testes-imports-e-mecanismos.md](ADR-043-auditoria-final-integral-testes-imports-e-mecanismos.md) |
| ADR-044 | MindControl: cérebro externo e mutação runtime segura | Ativo | 2026-02-20 | [ADR-044-mindcontrol-cerebro-externo-e-mutacao-runtime-segura.md](ADR-044-mindcontrol-cerebro-externo-e-mutacao-runtime-segura.md) |
| ADR-045 | Checkpoints seguros com safetensors, flush lazy e validação de shape | Ativo | 2026-02-20 | [ADR-045-checkpoints-seguros-com-safetensors-flush-lazy-e-shape-validation.md](ADR-045-checkpoints-seguros-com-safetensors-flush-lazy-e-shape-validation.md) |

## ADRs legados e superseded

| ID | Título | Status | Data | Arquivo |
| :--- | :--- | :--- | :--- | :--- |
| ADR-0001 | Import contract and release readiness | Legado | N/D | [legado/0001-import-contract-and-release-readiness.md](legado/0001-import-contract-and-release-readiness.md) |
| ADR-0040 | Conclusão do ciclo ISSUE e foco em execução | Legado | N/D | [legado/0040-conclusao-do-ciclo-issue-e-foco-em-execucao.md](legado/0040-conclusao-do-ciclo-issue-e-foco-em-execucao.md) |
| ADR-038 (duplicado) | Governança do `.fold/.mind`: auditoria de integridade e prompt operacional padronizado | Superseded | 2026-02-19 | [legado/ADR-038-governanca-fold-mind-auditoria-integridade-e-prompt-operacional.md](legado/ADR-038-governanca-fold-mind-auditoria-integridade-e-prompt-operacional.md) |
