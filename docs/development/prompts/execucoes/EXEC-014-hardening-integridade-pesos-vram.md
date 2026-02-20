# EXEC-014 — Hardening de integridade de pesos (VRAM)

## Plano executado

1. Implementar monitor de hash periódico em runtime.
2. Expor API pública para uso em pipelines de treino.
3. Cobrir com testes unitários.
4. Atualizar governança (ADR, ISSUE de documentação e HUB).

## Artefatos alterados

- `src/pyfolds/monitoring/health.py`
- `src/pyfolds/monitoring/__init__.py`
- `tests/unit/core/test_health_monitor.py`
- `docs/governance/adr/ADR-046-monitoramento-periodico-de-integridade-dos-pesos.md`
- `docs/governance/quality/issues/ISSUE-009-atualizacao-documentacao-v2-0-2.md`
- `docs/development/HUB_CONTROLE.md`
- `docs/development/execution_queue.csv`
- `docs/governance/adr/INDEX.md`
- `CHANGELOG.md`
