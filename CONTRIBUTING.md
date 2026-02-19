# Contributing to PyFolds

Este guia resume o fluxo de contribuição usado no repositório.

## 1) Antes de começar
- Leia `docs/development/HUB_CONTROLE.md`.
- Se for trabalho novo, registre/atualize a issue em `docs/development/execution_queue.csv`.
- Se houver decisão arquitetural, trate ADR correspondente em `docs/governance/adr/`.

## 2) Fluxo por tipo de mudança

### Código (`src/pyfolds/`)
1. Faça mudanças pequenas, rastreáveis e com testes.
2. Atualize documentação de API quando necessário.
3. Rode:

```bash
python -m compileall src/
python tools/check_api_docs.py --strict
python tools/check_links.py docs/ README.md
PYTHONPATH=src pytest tests/ -v
```

### Documentação (`docs/`)
1. Atualize índices/links quando necessário.
2. Rode:

```bash
python tools/check_links.py docs/ README.md
python tools/sync_hub.py --check
```

### ADR / Governança
1. Crie/atualize ADR em `docs/governance/adr/`.
2. Atualize `docs/governance/adr/INDEX.md`.
3. Sincronize fila/HUB quando houver impacto operacional.

## 3) Checklist mínimo para PR
- [ ] Escopo da mudança está claro.
- [ ] Validações locais executadas.
- [ ] Docs atualizadas quando aplicável.
- [ ] Issue/ADR referenciadas quando houver impacto.
- [ ] `docs/development/execution_queue.csv` atualizado (quando aplicável).

## 4) Referências internas
- Guia detalhado de desenvolvimento: `docs/development/CONTRIBUTING.md`
- Workflow operacional de issues: `docs/development/prompts/README.md`
- Workflow integrado (docs → código): `docs/development/WORKFLOW_INTEGRADO.md`
- Processo de release: `docs/development/release_process.md`
- Governança de prevenção de conflitos Git: `docs/governance/GIT_CONFLICT_PREVENTION.md`

## 5) Como criar uma nova Issue (Fluxo CRIAR → ANALISAR → EXECUTAR)
1. **CRIAR:** `python tools/create_issue_report.py --issue-id ISSUE-XXX --tema "..." --prioridade "Alta" --area "Core"`
2. **ANALISAR:** revisar requisitos no relatório criado em `docs/development/prompts/relatorios/`.
3. **EXECUTAR:** implementar código e testes vinculados.
4. **VALIDAR:** `python tools/validate_issue_format.py docs/development/prompts/relatorios/`.
5. **SINCRONIZAR:** `python tools/sync_hub_auto.py`.

Templates:
- `docs/development/templates/ISSUE-IA-TEMPLATE.md`
- `docs/development/templates/ISSUE-LOG-TEMPLATE.md`
