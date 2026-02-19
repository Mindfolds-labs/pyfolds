# Templates de Relatórios

## Como criar nova issue
Use `python tools/create_issue_report.py --issue-id ISSUE-XXX --tema "..." --prioridade "Alta" --area "Core"`.

## Como executar batch
Use `python tools/batch_create_issues.py --config docs/development/batch_issues.json`.

## Como validar formato
Use `python tools/validate_issue_format.py docs/development/prompts/relatorios/`.

## Padrões de nomeação
- Relatórios: `ISSUE-XXX-slug.md`
- Logs: `ISSUE-XXX-create.log.json`

## Links úteis
- `./ISSUE-IA-TEMPLATE.md`
- `./ISSUE-LOG-TEMPLATE.md`
