# EXEC-038 — auditoria de prontidão para publicação no PyPI

## Status
🟢 Concluída

## Escopo executado
- Auditoria de estrutura e artefatos obrigatórios para release PyPI.
- Validação prática de build e empacotamento.
- Validação de distribuição com `twine`.
- Execução de testes para baseline de estabilidade.
- Registro de governança completo com ISSUE/ADR/fila/HUB.

## Comandos executados
- `python -m build`
- `twine check dist/*`
- `PYTHONPATH=src pytest -q`
- `python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-038-auditoria-prontidao-publicacao-pypi.md`
- `python tools/sync_hub.py`
- `python tools/sync_hub.py --check`
- `python tools/check_issue_links.py docs/development/prompts/relatorios`

## Resultado consolidado
- Distribuições sdist e wheel geradas com sucesso.
- `twine check` aprovado para todos os artefatos em `dist/`.
- Suite de testes principal aprovada (232 passed).
- Foram detectados avisos de compatibilidade futura no `setuptools` (licença/classifiers/keywords), registrados como dívida técnica de governança.

## Prompt pronto para reutilização no Codex
```text
Objetivo: auditar prontidão para release no PyPI e consolidar governança de entrega.

1) Executar:
   python -m build
2) Validar distribuição:
   twine check dist/*
3) Validar estabilidade:
   PYTHONPATH=src pytest -q
4) Registrar governança:
   - ISSUE-NNN no formato validado
   - EXEC-NNN correspondente
   - ADR-039 correspondente com decisões de release
   - atualizar docs/development/execution_queue.csv
   - rodar python tools/sync_hub.py
5) Validar:
   python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-NNN-*.md
   python tools/sync_hub.py --check
   python tools/check_issue_links.py docs/development/prompts/relatorios
```
