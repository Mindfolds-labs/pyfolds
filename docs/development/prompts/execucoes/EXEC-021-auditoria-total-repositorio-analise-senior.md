# EXEC-021 — Auditoria total do repositório com análise sênior (sem execução)

## Status
🟡 Planejada (escopo analítico)

## Diretriz de execução
Esta EXEC formaliza que a frente `ISSUE-021` deve ser conduzida como **análise completa do repositório**, com profundidade técnica de engenharia sênior, **sem implementação de mudanças funcionais de produto** neste ciclo.

## Escopo operacional da EXEC
1. Mapear estado atual de código, documentação, testes, CI e governança.
2. Consolidar achados por criticidade (crítico/médio/baixo).
3. Definir plano incremental com critérios objetivos de aceite.
4. Registrar evidências no fluxo oficial: ISSUE → EXEC → CSV → HUB.

## Restrições obrigatórias
- Não alterar comportamento de `src/pyfolds/**` neste ciclo.
- Não executar refatorações estruturais fora de governança documental.
- Não marcar como concluída sem validações de consistência do HUB.

## Validações previstas
- `python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-021-auditoria-total-repositorio-analise-senior.md`
- `python tools/sync_hub.py`
- `python tools/sync_hub.py --check`
- `python tools/check_issue_links.py docs/development/prompts/relatorios`
