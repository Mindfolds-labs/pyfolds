# EXEC-036 — auditoria do fluxo `.fold/.mind` + governança completa

## Status
🟢 Concluída

## Escopo executado
- Revisão técnica de `foldio.py` com foco em integridade, segurança e rastreabilidade.
- Verificação de regressão em testes de serialização/corrupção.
- Ajuste corretivo de tratamento de exceções na validação de assinatura digital.
- Registro completo de governança: ISSUE-036, ADR-038, `execution_queue.csv` e `HUB_CONTROLE.md`.

## Comandos executados
- `PYTHONPATH=src pytest -q tests/unit/serialization/test_foldio.py tests/test_fold_corruption.py tests/test_corruption_detection.py tests/test_concurrent_reads.py`
- `PYTHONPATH=src python -m py_compile src/pyfolds/serialization/foldio.py tests/unit/serialization/test_foldio.py`
- `python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-036-auditoria-fluxo-fold-mind-governanca.md`
- `python tools/sync_hub.py`
- `python tools/sync_hub.py --check`
- `python tools/check_issue_links.py docs/development/prompts/relatorios`

## Resultado consolidado
- Fluxo `.fold/.mind` validado sem erro crítico funcional.
- Testes focados aprovados (com skip esperado para cenário sem `cryptography`).
- Governança sincronizada com registro de ISSUE-036 e card atualizado no HUB.

## Prompt pronto para reutilização no Codex
```text
Objetivo: Auditar o formato .fold/.mind, validar lógica de serialização e segurança, e consolidar governança.

1) Rodar:
   PYTHONPATH=src pytest -q tests/unit/serialization/test_foldio.py tests/test_fold_corruption.py tests/test_corruption_detection.py tests/test_concurrent_reads.py
2) Revisar src/pyfolds/serialization/foldio.py em:
   - validação de header/index
   - checks CRC32C/SHA256
   - desserialização torch com weights_only=True
   - assinatura opcional e erros padronizados para FoldSecurityError
3) Se necessário, ajustar testes em tests/unit/serialization/test_foldio.py.
4) Gerar trilha de governança:
   - ISSUE-NNN no formato validado
   - EXEC-NNN correspondente
   - atualizar execution_queue.csv
   - rodar python tools/sync_hub.py
   - garantir alteração de docs/development/HUB_CONTROLE.md
5) Validar:
   python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-NNN-*.md
   python tools/sync_hub.py --check
   python tools/check_issue_links.py docs/development/prompts/relatorios
```
