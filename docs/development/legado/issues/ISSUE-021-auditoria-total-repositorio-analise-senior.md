# ISSUE-021: auditoria total do repositório com análise sênior (sem execução de mudanças)

## Metadados
- id: ISSUE-021
- tipo: GOVERNANCE
- titulo: Auditoria total do repositório com análise sênior (sem execução)
- criado_em: 2026-02-17
- owner: Codex
- status: TODO

## 1. Objetivo
Conduzir uma análise completa e técnica de todo o repositório PyFolds, no nível de engenharia sênior, com foco em diagnóstico, rastreabilidade e plano de melhoria, sem executar alterações funcionais em código de produção neste ciclo.

## 2. Escopo

### 2.1 Inclui:
- Auditoria integral dos diretórios `src/`, `tests/`, `docs/`, `.github/`, `tools/`, `examples/` e arquivos de governança na raiz.
- Revisão de conformidade documental e de processo conforme fluxo oficial de prompts.
- Identificação de gaps críticos, médios e baixos com priorização clara e plano incremental.
- Proposta de ações para qualidade técnica, consistência de documentação e robustez de CI.
- Registro formal em ISSUE/EXEC/CSV/HUB com numeração regular e rastreável.

### 2.2 Exclui:
- Implementação de correções de produto em `src/pyfolds/**`.
- Mudanças invasivas de arquitetura ou refatorações amplas neste ciclo.
- Fechamento de débitos técnicos além da elaboração de plano e critérios de aceite.

## 3. Artefatos Gerados
- `docs/development/prompts/relatorios/ISSUE-021-auditoria-total-repositorio-analise-senior.md`
- `docs/development/prompts/execucoes/EXEC-021-auditoria-total-repositorio-analise-senior.md`
- `docs/development/prompts/README.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`

## 4. Riscos
- Risco: confundir ciclo de análise com ciclo de execução e gerar mudanças fora do escopo.
  Mitigação: explicitar em ISSUE e EXEC que esta frente é exclusivamente diagnóstica.
- Risco: deriva de numeração entre ISSUE/EXEC/CSV/HUB.
  Mitigação: seguir algoritmo obrigatório, sincronizar HUB e validar com `--check`.
- Risco: relatório amplo sem priorização acionável.
  Mitigação: classificar achados por criticidade e associar plano em fases.

## 5. Critérios de Aceite
- ISSUE no formato validável por `tools/validate_issue_format.py`.
- Existência do par `ISSUE-021` + `EXEC-021` com o mesmo slug temático.
- Registro da `ISSUE-021` em `docs/development/execution_queue.csv`.
- Execução de `python tools/sync_hub.py` e alteração correspondente em `docs/development/HUB_CONTROLE.md` no mesmo commit.
- Validações obrigatórias executadas com sucesso:
  - `python tools/sync_hub.py --check`
  - `python tools/check_issue_links.py docs/development/prompts/relatorios`

## 6. PROMPT:EXECUTAR
```yaml
issue_id: "ISSUE-021"
tipo: "GOVERNANCE"
titulo: "Auditoria total do repositório com análise sênior (sem execução)"

passos_obrigatorios:
  - "Ler docs/development/execution_queue.csv"
  - "Confirmar próximo ISSUE-NNN regular"
  - "Manter escopo estritamente analítico (sem execução de mudanças de produto)"
  - "Criar/atualizar ISSUE-021 e EXEC-021"
  - "Registrar ISSUE-021 no execution_queue.csv"
  - "Rodar python tools/sync_hub.py"
  - "Garantir alteração de docs/development/HUB_CONTROLE.md no mesmo commit"

validacao:
  - "python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-021-auditoria-total-repositorio-analise-senior.md"
  - "python tools/sync_hub.py --check"
  - "python tools/check_issue_links.py docs/development/prompts/relatorios"
```
