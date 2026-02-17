# EXEC-018 — Padronização de relatórios e sync HUB obrigatório

## Status
✅ Concluída

## Resumo da execução
- Revisado o fluxo de prompts e removidas duplicidades/inconsistências no `docs/development/prompts/README.md`.
- Formalizado fluxo obrigatório: atualização de CSV + execução de `sync_hub.py` + atualização de `HUB_CONTROLE.md` no mesmo commit.
- Padronizado `ISSUE-000-template.md` para manter compatibilidade com o validador e adicionar referência canônica ao modelo de análise da `ISSUE-003`.
- Atualizado `relatorios/README.md` para deixar explícito o vínculo entre formato validado e padrão de relatório técnico.
- Registrada `ISSUE-018` no CSV e sincronizado HUB.

## Erro encontrado (causa raiz)
Desde ciclos após a `ISSUE-003`, o processo ficou com instruções dispersas e parcialmente duplicadas:
1. README de prompts com blocos repetidos e sem gate explícito de "mesmo commit" para HUB.
2. Template anterior não refletia simultaneamente o formato exigido por `validate_issue_format.py` e o padrão analítico de `ISSUE-003`.
3. Existência de poucos `EXEC-*` frente ao histórico de `ISSUE-*`, reduzindo rastreabilidade operacional.

## Correção aplicada
- Fluxo consolidado em um único checklist operacional no README de prompts.
- Template atualizado com seções obrigatórias do validador + apêndice canônico de análise.
- Entrega desta execução com ISSUE+EXEC+CSV+HUB sincronizados.

## Validações executadas
- `python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-018-padronizacao-relatorios-sync-hub-obrigatorio.md`
- `python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-000-template.md`
- `python tools/sync_hub.py`
- `python tools/sync_hub.py --check`
- `python tools/check_issue_links.py docs/development/prompts/relatorios`
