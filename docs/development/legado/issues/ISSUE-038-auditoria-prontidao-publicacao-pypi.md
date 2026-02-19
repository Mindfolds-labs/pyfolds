# ISSUE-038: auditoria de prontidão para publicação no pypi

## Metadados
- id: ISSUE-038
- tipo: GOVERNANCE
- titulo: Auditoria sênior de prontidão para release no PyPI com checklist técnico e evidências de build
- criado_em: 2026-02-18
- owner: Codex
- status: DONE

## 1. Objetivo
Executar uma auditoria de engenharia sênior para verificar se o pacote `pyfolds` está pronto para publicação no PyPI, consolidando evidências práticas de build/publicação e definindo plano de correção para gaps de conformidade.

## 2. Escopo

### 2.1 Inclui:
- Auditoria da estrutura do projeto (`src/`, `tests/`, `docs/`, `scripts/`) e dos artefatos de empacotamento.
- Validação prática de build e validação de metadados de distribuição (`python -m build`, `twine check dist/*`).
- Execução de suíte de testes para baseline de estabilidade antes de release.
- Geração da trilha completa de governança com ISSUE-038, EXEC-038, ADR-039, atualização do CSV e sincronização do HUB.

### 2.2 Exclui:
- Publicação real no PyPI/TestPyPI (sem credenciais neste fluxo).
- Refatorações profundas de arquitetura fora de gaps diretamente ligados à prontidão de release.

## 3. Artefatos Gerados
- `docs/development/prompts/relatorios/ISSUE-038-auditoria-prontidao-publicacao-pypi.md`
- `docs/development/prompts/execucoes/EXEC-038-auditoria-prontidao-publicacao-pypi.md`
- `docs/governance/adr/ADR-039-auditoria-prontidao-publicacao-pypi.md`
- `docs/governance/adr/INDEX.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`

## 4. Riscos
- Risco: publicação falhar no PyPI por inconsistência entre `pyproject.toml` e `setup.cfg`.
  Mitigação: manter metadados canônicos no `[project]` e remover duplicidade legacy em próxima issue de hardening.
- Risco: avisos de depreciação evoluírem para erro em versões futuras de `setuptools`.
  Mitigação: migrar `license` para expressão SPDX string e mover `classifiers/keywords` para `pyproject.toml`.
- Risco: regressão funcional não detectada antes da release.
  Mitigação: manter execução obrigatória de `pytest`, `build` e `twine check` no gate de release.

## 5. Critérios de Aceite
- ISSUE em conformidade com `tools/validate_issue_format.py`.
- EXEC com passos executados e validações registradas.
- Registro no `execution_queue.csv`.
- `python tools/sync_hub.py` executado.
- `HUB_CONTROLE.md` alterado no mesmo commit.
- Evidências de release registradas (`python -m build`, `twine check dist/*`, `pytest`).

## 6. PROMPT:EXECUTAR
```yaml
issue_id: "ISSUE-038"
tipo: "GOVERNANCE"
titulo: "Auditoria sênior de prontidão PyPI com checklist operacional"

passos_obrigatorios:
  - "Revisar pyproject.toml, setup.cfg, MANIFEST.in e artefatos obrigatórios da raiz"
  - "Executar python -m build"
  - "Executar twine check dist/*"
  - "Executar PYTHONPATH=src pytest -q"
  - "Gerar ISSUE-038 e EXEC-038 com evidências"
  - "Atualizar/registrar ADR-039 com decisões de governança de release"
  - "Registrar ISSUE-038 em docs/development/execution_queue.csv"
  - "Rodar python tools/sync_hub.py"
  - "Garantir alteração de docs/development/HUB_CONTROLE.md no mesmo commit"

validacao:
  - "python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-038-auditoria-prontidao-publicacao-pypi.md"
  - "python tools/sync_hub.py --check"
  - "python tools/check_issue_links.py docs/development/prompts/relatorios"
```

---

## Apêndice A — Relatório técnico (modelo ISSUE-003)

### A.1 Sumário executivo
O pacote está funcionalmente próximo de pronto para PyPI: build e validação de distribuição passaram com sucesso, e a suíte principal de testes foi aprovada. Foram identificados gaps de governança de metadados que não bloqueiam a publicação imediata, mas devem entrar no próximo ciclo para reduzir risco futuro.

### A.2 Diagnóstico por checklist
- **Estrutura de projeto:** conforme (`src/pyfolds/`, `tests/`, `docs/`, `scripts/`).
- **Arquivos essenciais:** presentes (`pyproject.toml`, `README.md`, `LICENSE`, `CHANGELOG.md`, `MANIFEST.in`).
- **Versionamento e empacotamento:** `1.0.1` em formato SemVer; build e `twine check` aprovados.
- **Qualidade e testes:** suíte `pytest` passou (232 testes), com 2 warnings não bloqueantes.
- **Conformidade PyPI:** metadados principais ok, porém há avisos de depreciação e duplicidade entre `pyproject.toml` e `setup.cfg`.

### A.3 Evidências executadas
- `python -m build`
- `twine check dist/*`
- `PYTHONPATH=src pytest -q`

### A.4 Plano de ação recomendado (próxima issue)
1. Migrar `project.license` para string SPDX e remover classifier de licença legado.
2. Declarar `classifiers` e `keywords` diretamente em `pyproject.toml`.
3. Minimizar duplicidade de metadados entre `pyproject.toml` e `setup.cfg`.
4. Adicionar gate CI dedicado de release (`build + twine check + smoke install`).
