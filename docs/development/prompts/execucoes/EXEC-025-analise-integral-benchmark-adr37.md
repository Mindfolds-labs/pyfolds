# EXEC-025 — análise integral do pyfolds + benchmark refresh

## Status
🟢 Concluída

## Escopo executado
- Compilação integral dos módulos em `src/`.
- Execução da suíte principal (`200 passed`).
- Atualização dos artefatos de benchmark em `docs/assets/`.
- Registro da decisão de governança em ADR-037.

## Comandos de validação
- `python -m compileall src`
- `PYTHONPATH=src pytest -q`
- `python scripts/run_benchmarks.py --output docs/assets/benchmarks_results.json`
- `python scripts/generate_benchmarks_doc.py --input docs/assets/benchmarks_results.json --output docs/assets/BENCHMARKS.md`
- `python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-025-analise-integral-benchmark-adr37.md`
- `python tools/sync_hub.py`
- `python tools/sync_hub.py --check`
- `python tools/check_issue_links.py docs/development/prompts/relatorios`

## Resultado consolidado
- Execução principal estável, sem falhas de teste.
- Warnings não-bloqueantes observados:
  - `PytestUnknownMarkWarning` para marca `performance` não registrada.
  - `DeprecationWarning` para `datetime.utcnow()` em checkpoint versionado.
  - `RuntimeWarning` esperado em teste de limpeza forçada de `mmap`.
- Benchmark atualizado com amostras atuais e compressão por fallback `zlib(level=6)` (sem `zstd` no ambiente).

## Aprovação operacional
- ISSUE-025 executada de forma direta, com evidências completas e pronta para aprovação humana de fechamento.
