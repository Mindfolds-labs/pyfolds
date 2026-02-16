# ADR Index — Governança de arquitetura e formato

## Mapa de dependências

```text
ADR-001 (Formalização da Spec)
  ├─> ADR-002 (Container .fold/.mind)
  ├─> ADR-004 (Versionamento e compatibilidade)
  └─> ADR-007 (Testes de corrupção e recuperação)

ADR-003 (Hardening inicial)
  ├─> ADR-006 (Invariantes e estabilidade numérica)
  └─> ADR-007 (Testes de corrupção e recuperação)

ADR-004 (Versionamento)
  └─> ADR-009 (Governança de release)

ADR-008 (Benchmark de performance)
  └─> ADR-009 (Governança operacional e release)
```

## Lista de ADRs

- [ADR-001 — Formalização da especificação `.fold/.mind`](./ADR-001-formalizacao-spec-fold-mind.md)
- [ADR-002 — Container `.fold/.mind` para checkpoints científicos](./ADR-002-container-fold-mind.md)
- [ADR-003 — Correção de regressões e hardening inicial](./ADR-003-correcao-regressoes-hardening.md)
- [ADR-004 — Política de versionamento e compatibilidade do formato](./ADR-004-politica-versionamento-compatibilidade.md)
- [ADR-006 — Invariantes e estabilidade numérica como gate de qualidade](./ADR-006-invariantes-estabilidade-numerica.md)
- [ADR-007 — Testes de corrupção e recuperação segura](./ADR-007-teste-corrupcao-recuperacao-segura.md)
- [ADR-008 — Benchmark contínuo de serialização e escala](./ADR-008-benchmark-performance-serializacao.md)
- [ADR-009 — Governança operacional e checklist de release](./ADR-009-governanca-operacao-release.md)

## Notas

- ADR-005 está reservado para tema futuro e não faz parte do conjunto atual.
- ADRs antigos em `docs/developments/adr/` devem ser considerados legado documental.
