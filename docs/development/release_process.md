# Release Process (PyFolds)

> Integrado ao fluxo de issues em `docs/development/HUB_CONTROLE.md`.

## Objetivo
Formalizar o processo de release com critérios técnicos e de governança verificáveis.

## Pré-condições
1. Issues do escopo da release com status apropriado em `docs/development/execution_queue.csv`.
2. HUB sincronizado (`python tools/sync_hub.py --check`).
3. ADRs novas/impactadas registradas no índice (`docs/governance/adr/INDEX.md`) quando aplicável.
4. `CHANGELOG.md` atualizado com mudanças externas relevantes.

## Fluxo de release

### Fase 1 — Preparação
1. Confirmar escopo da release (features/fixes/docs).
2. Consolidar evidências de PRs aprovadas.
3. Atualizar `CHANGELOG.md`.

### Fase 2 — Validação técnica
Executar:

```bash
python -m compileall src/
python tools/check_api_docs.py --strict
python tools/check_links.py docs/ README.md
PYTHONPATH=src pytest tests/ -v --maxfail=1
python tools/sync_hub.py --check
```

### Fase 3 — Versionamento e publicação
1. Definir versão semântica.
2. Criar tag (`vX.Y.Z`).
3. Publicar release com notas alinhadas ao `CHANGELOG.md`.

### Fase 4 — Pós-release
1. Registrar conclusão de sprint/entregas no fluxo de governança.
2. Planejar próxima janela de execução.

## Checklist de release

### A. Técnicos
- [ ] `python -m compileall src/`.
- [ ] `python tools/check_api_docs.py --strict`.
- [ ] `python tools/check_links.py docs/ README.md`.
- [ ] `PYTHONPATH=src pytest tests/ -v --maxfail=1`.
- [ ] `python tools/sync_hub.py --check`.

### B. Governança

### B.1 Pre-merge governance gates
- [ ] `python tools/check_queue_governance.py --base-ref main`.
- [ ] `python tools/sync_hub.py --check`.
- [ ] `docs/development/execution_queue.csv` e `docs/development/HUB_CONTROLE.md` alterados em conjunto quando houver mudança de fila.
- [ ] Regras do documento canônico validadas: `docs/governance/GIT_CONFLICT_PREVENTION.md`.

- [ ] `CHANGELOG.md` atualizado.
- [ ] `docs/development/execution_queue.csv` refletindo escopo da release.
- [ ] `docs/development/HUB_CONTROLE.md` consistente.
- [ ] ADRs impactadas atualizadas (se houver).

### C. Publicação
- [ ] Tag semântica criada.
- [ ] Release publicada com notas.
- [ ] Evidências arquivadas (PRs, checks, tag).

## Evidências mínimas
- Commit/tag da release.
- Comandos executados e respectivos resultados.
- Referências às PRs aprovadas no ciclo.
