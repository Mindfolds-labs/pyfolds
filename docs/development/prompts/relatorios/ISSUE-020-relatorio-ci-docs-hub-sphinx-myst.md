# ISSUE-020: relatório ci docs hub e correções para sphinx/myst

## Metadados
- id: ISSUE-020
- tipo: GOVERNANCE
- titulo: Relatório CI Docs Hub e correções para Sphinx/MyST
- criado_em: 2026-02-17
- owner: Codex
- status: TODO

## 1. Objetivo
Gerar um relatório técnico padronizado para explicar a causa da falha recorrente do job `docs-hub-quality` e definir um plano mínimo de correção para Sphinx/MyST (incluindo dependência de PlantUML), com registro completo no HUB.

## 2. Escopo

### 2.1 Inclui:
- Auditar links e referências em `docs/`, com foco em `docs/DEVELOPMENT_HUB.md` e `docs/index.md`.
- Mapear ocorrências de blocos Mermaid/PlantUML e explicitar pré-requisitos de execução no CI.
- Levantar arquivos com links internos quebrados que podem gerar `myst.xref_missing`.
- Produzir relatório EXEC com causas-raiz, plano mínimo de patch e comandos de validação.
- Registrar ISSUE na `execution_queue.csv` e sincronizar `HUB_CONTROLE.md`.

### 2.2 Exclui:
- Refatoração arquitetural do projeto.
- Alterações no código-fonte de produção em `src/`.
- Reescrita total de ADRs históricos (apenas diagnóstico e patch mínimo recomendado).

## 3. Artefatos Gerados
- `docs/development/prompts/relatorios/ISSUE-020-relatorio-ci-docs-hub-sphinx-myst.md`
- `docs/development/prompts/logs/EXEC-020-relatorio-ci-docs-hub.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`

## 4. Riscos
- Risco: CI continuar falhando por warnings tratados como erro (`-W`) no Sphinx.
  Mitigação: corrigir links/referências internas quebradas e reduzir baseline de warnings antes de endurecer gate.
- Risco: falha por ausência de runtime PlantUML/Java.
  Mitigação: declarar instalação de PlantUML no workflow (ou remover extensão quando não usada no build).
- Risco: divergência entre queue CSV e card do HUB.
  Mitigação: executar `tools/sync_hub.py` no mesmo commit e validar com `--check`.

## 5. Critérios de Aceite
- ISSUE em conformidade com `tools/validate_issue_format.py`.
- EXEC com diagnóstico técnico e plano mínimo de correção para CI docs.
- Registro no `docs/development/execution_queue.csv`.
- `python tools/sync_hub.py` executado no mesmo ciclo.
- `docs/development/HUB_CONTROLE.md` alterado no mesmo commit.
- Validações obrigatórias executadas e registradas.

## 6. PROMPT:EXECUTAR
```yaml
issue_id: "ISSUE-020"
tipo: "GOVERNANCE"
titulo: "Relatório CI Docs Hub e correções para Sphinx/MyST"

passos_obrigatorios:
  - "Ler docs/development/execution_queue.csv"
  - "Descobrir próximo ISSUE-NNN"
  - "Criar ISSUE-[NNN]-[slug].md"
  - "Criar EXEC-[NNN]-[slug].md"
  - "Registrar ISSUE no execution_queue.csv"
  - "Rodar python tools/sync_hub.py"
  - "Garantir alteração de docs/development/HUB_CONTROLE.md no mesmo commit"

validacao:
  - "python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-[NNN]-[slug].md"
  - "python tools/sync_hub.py --check"
  - "python tools/check_issue_links.py docs/development/prompts/relatorios"
```
