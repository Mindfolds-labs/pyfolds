# ISSUE-000: [titulo-em-minusculas]

## Metadados
- id: ISSUE-000
- tipo: [CODE|DOCS|TEST|ADR|GOVERNANCE]
- titulo: [Título objetivo]
- criado_em: [YYYY-MM-DD]
- owner: [Nome]
- status: TODO

## 1. Objetivo
Descrever claramente o problema, a motivação e o resultado esperado.

## 2. Escopo

### 2.1 Inclui:
- Item 1
- Item 2

### 2.2 Exclui:
- Item fora de escopo

## 3. Artefatos Gerados
- `docs/development/prompts/relatorios/ISSUE-000-[slug].md`
- `docs/development/prompts/execucoes/EXEC-000-[slug].md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`

## 4. Riscos
- Risco: [descrição]
  Mitigação: [ação]

## 5. Critérios de Aceite
- ISSUE em conformidade com `tools/validate_issue_format.py`
- EXEC com passos executados e validações registradas
- Registro no `execution_queue.csv`
- `python tools/sync_hub.py` executado
- `HUB_CONTROLE.md` alterado no mesmo commit

## 6. PROMPT:EXECUTAR
```yaml
issue_id: "ISSUE-000"
tipo: "[CODE|DOCS|TEST|ADR|GOVERNANCE]"
titulo: "[Título objetivo]"

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

---

## Apêndice A — Estrutura canônica de relatório técnico (referência ISSUE-003)
Para manter consistência histórica, o corpo analítico deve espelhar `ISSUE-003-auditoria-completa.md` com:

1. Sumário executivo
2. Diagnóstico e análise
3. Artefatos atualizados
4. Execução técnica
5. Riscos, restrições e mitigações
6. Critérios de aceite e status
