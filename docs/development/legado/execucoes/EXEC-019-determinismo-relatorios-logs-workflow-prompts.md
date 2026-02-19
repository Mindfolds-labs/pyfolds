# EXEC-019 — Determinismo de relatórios e logs no workflow de prompts

## Status
✅ Concluída

## Diagnóstico do fluxo atual
### Erro encontrado 1 — Variação de conteúdo entre relatórios e logs
- **Sintoma:** alguns ciclos têm ISSUE detalhada, mas EXEC curto ou ausente, dificultando reprodução por outro agente.
- **Impacto:** perda de previsibilidade na auditoria e na revisão de PR.
- **Correção proposta:** exigir em toda EXEC os blocos mínimos: "Erro encontrado", "Impacto", "Correção aplicada", "Validação".

### Erro encontrado 2 — Atualização parcial de governança
- **Sintoma:** mudanças podem ficar apenas no CSV sem card detalhado coerente no HUB.
- **Impacto:** leitura fragmentada do histórico de execução.
- **Correção proposta:** checklist operacional exigindo sincronização HUB e confirmação de link de relatório no card.

### Erro encontrado 3 — Ambiguidade para manipulação por qualquer agente
- **Sintoma:** falta de instrução explícita e padronizada de como descrever arquivos alterados e causa-raiz.
- **Impacto:** baixa determinização na execução entre agentes diferentes.
- **Correção proposta:** reforçar padrão textual no relatório e execução com campos fixos por item (arquivo → erro → ação corretiva).

## Arquivos analisados e orientação de correção
1. `docs/development/prompts/README.md`
   - Ajuste recomendado: manter seção obrigatória de fluxo, com sequência rígida ISSUE/EXEC/CSV/HUB/validação.
2. `docs/development/prompts/relatorios/README.md`
   - Ajuste recomendado: explicitar que relatório deve conter diagnóstico técnico reproduzível.
3. `docs/development/prompts/relatorios/ISSUE-000-template.md`
   - Ajuste recomendado: preservar estrutura validável e exigir detalhamento de erro + mitigação.
4. `docs/development/HUB_CONTROLE.md`
   - Ajuste recomendado: garantir tabela sincronizada e card com link de relatório para cada ISSUE relevante.

## Padrão de detalhe mínimo (proposto)
Para cada erro relatado no ciclo:
1. **Arquivo afetado**
2. **Erro encontrado**
3. **Impacto no workflow**
4. **Correção aplicada/proposta**
5. **Comando de validação executado**

## Validações executadas
- `python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-019-determinismo-relatorios-logs-workflow-prompts.md`
- `python tools/sync_hub.py`
- `python tools/sync_hub.py --check`
- `python tools/check_issue_links.py docs/development/prompts/relatorios`
