# EXEC-036 — auditoria fold/mind, integridade lógica e governança operacional

## Status
🟢 Concluída

## Tarefa
Executar auditoria técnica do formato `.fold/.mind`, corrigir comportamento de erro de segurança na validação de assinatura digital e consolidar trilha de governança completa (ADR + ISSUE/EXEC + CSV/HUB).

## Contexto
A demanda solicitou confirmação objetiva sobre robustez do formato `.fold/.mind`, identificação de possíveis erros lógicos e entrega de prompt operacional para futuras execuções no Codex, com documentação em padrão de governança do projeto.

## Passos executados
1. Revisão de `foldio.py` com foco em verificação de assinatura e semântica de erro.
2. Ajuste no fluxo `load_fold_or_mind` para encapsular falhas de parsing/verificação de chave pública em `FoldSecurityError`.
3. Atualização do teste correspondente para validar erro de segurança explícito.
4. Criação do ADR-038 e indexação no `INDEX.md`.
5. Criação de ISSUE-036 no formato validável e inclusão de prompt operacional.
6. Atualização da fila oficial e sincronização do HUB.

## Validações
- `PYTHONPATH=src pytest -q tests/unit/serialization/test_foldio.py`
- `PYTHONPATH=src python -m py_compile src/pyfolds/serialization/foldio.py tests/unit/serialization/test_foldio.py`
- `python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-036-auditoria-fold-mind-integridade-governanca.md`
- `python tools/sync_hub.py`
- `python tools/sync_hub.py --check`
- `python tools/check_issue_links.py docs/development/prompts/relatorios`

## Atualização final de governança
- ISSUE-036 registrada em `docs/development/execution_queue.csv`.
- HUB sincronizado com `tools/sync_hub.py`.
- ADR-038 adicionado e referenciado no índice.
