# ADR Legado

Página de navegação para ADRs históricas.

## Como obter o próximo ADR

Use o scanner automático:

```bash
python -c "from tools.id_registry import next_adr_id; print(next_adr_id())"
```

Regra:
- sem histórico: `ADR-0001`;
- com histórico: próximo número disponível (`max + 1`).

## Arquivos

- [0001-import-contract-and-release-readiness.md](./0001-import-contract-and-release-readiness.md)
- [0040-conclusao-do-ciclo-issue-e-foco-em-execucao.md](./0040-conclusao-do-ciclo-issue-e-foco-em-execucao.md)
- [ADR-038-governanca-fold-mind-auditoria-integridade-e-prompt-operacional.md](./ADR-038-governanca-fold-mind-auditoria-integridade-e-prompt-operacional.md)
