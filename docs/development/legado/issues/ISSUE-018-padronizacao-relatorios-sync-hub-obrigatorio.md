# ISSUE-018: padronizacao-relatorios-sync-hub-obrigatorio

## Metadados
- id: ISSUE-018
- tipo: GOVERNANCE
- titulo: Padronização de relatórios e obrigatoriedade de sync HUB
- criado_em: 2026-02-17
- owner: Codex
- status: DONE

## 1. Objetivo
Padronizar o fluxo documental de prompts para eliminar inconsistências entre relatório, fila de execução e HUB, reforçando a obrigatoriedade de sincronização via `tools/sync_hub.py` no mesmo commit.

## 2. Escopo

### 2.1 Inclui:
- Revisão dos READMEs relevantes de documentação e prompts para identificar conflitos de orientação.
- Diagnóstico do desvio operacional recorrente desde o ciclo pós `ISSUE-003` (fila atualizada sem sincronização explícita do HUB no momento da execução).
- Atualização do guia `docs/development/prompts/README.md` com seção de fluxo obrigatório e gate explícito de sync HUB.
- Ajuste do template base `ISSUE-000-template.md` para manter compatibilidade com `validate_issue_format.py` e incluir apêndice canônico de relatório inspirado em `ISSUE-003`.
- Geração de `EXEC-018` e registro da issue no `execution_queue.csv`.

### 2.2 Exclui:
- Refatorações estruturais de código em `src/`.
- Mudanças de lógica em `tools/validate_issue_format.py`.
- Mudanças fora de governança documental e fluxo de prompts.

## 3. Artefatos Gerados
- `docs/development/prompts/README.md`
- `docs/development/prompts/relatorios/README.md`
- `docs/development/prompts/relatorios/ISSUE-000-template.md`
- `docs/development/prompts/relatorios/ISSUE-018-padronizacao-relatorios-sync-hub-obrigatorio.md`
- `docs/development/prompts/execucoes/EXEC-018-padronizacao-relatorios-sync-hub-obrigatorio.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`

## 4. Riscos
- Risco: confusão entre o formato canônico de análise (ISSUE-003) e o formato validado por regex do `validate_issue_format.py`.
  Mitigação: documentar explicitamente os dois níveis no README de prompts e no template base.
- Risco: regressão operacional com atualização de CSV sem atualização de HUB no mesmo commit.
  Mitigação: checklist obrigatório com `sync_hub.py`, verificação de diff e validação `sync_hub.py --check`.

## 5. Critérios de Aceite
- README de prompts com seção explícita "Fluxo obrigatório de execução" incluindo sync HUB e atualização no mesmo commit.
- Template `ISSUE-000` compatível com validador e com seção de referência ao padrão canônico de `ISSUE-003`.
- `EXEC-018` criado com evidências do fluxo executado.
- `execution_queue.csv` atualizado com `ISSUE-018`.
- `docs/development/HUB_CONTROLE.md` alterado após `python tools/sync_hub.py`.

## 6. PROMPT:EXECUTAR
```yaml
issue_id: "ISSUE-018"
tipo: "GOVERNANCE"
titulo: "Padronização de relatórios e obrigatoriedade de sync HUB"

passos_obrigatorios:
  - "Ler docs/development/execution_queue.csv"
  - "Descobrir próximo ISSUE-NNN regular"
  - "Criar ISSUE-018-padronizacao-relatorios-sync-hub-obrigatorio.md"
  - "Criar EXEC-018-padronizacao-relatorios-sync-hub-obrigatorio.md"
  - "Registrar ISSUE-018 no execution_queue.csv"
  - "Atualizar docs/development/prompts/README.md com fluxo obrigatório"
  - "Rodar python tools/sync_hub.py"
  - "Garantir alteração de docs/development/HUB_CONTROLE.md no mesmo commit"

validacao:
  - "python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-018-padronizacao-relatorios-sync-hub-obrigatorio.md"
  - "python tools/sync_hub.py --check"
  - "python tools/check_issue_links.py docs/development/prompts/relatorios"
```
