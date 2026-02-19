# Guia de Automação do HUB

## Scripts principais
- `tools/create_issue_report.py`: cria issue com YAML + checksum.
- `tools/batch_create_issues.py`: cria várias issues a partir de JSON.
- `tools/sync_hub_auto.py`: sincroniza cards do HUB.
- `tools/validate_issue_format.py`: valida estrutura, YAML e links.
- `tools/generate_html_dashboard.py`: gera dashboard HTML.

## Fluxo recomendado
1. Criar issue: `python tools/create_issue_report.py ...`
2. Validar: `python tools/validate_issue_format.py docs/development/prompts/relatorios/`
3. Sincronizar HUB: `python tools/sync_hub_auto.py`
4. Gerar dashboard: `python tools/generate_html_dashboard.py`

## Troubleshooting
- Erro de YAML: revise frontmatter entre `---`.
- Link quebrado: ajuste links relativos no relatório.
- Duplicata de ID: use nova numeração de issue.
