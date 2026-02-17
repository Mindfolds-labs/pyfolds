# ISSUE-025: análise integral do pyfolds + atualização de benchmark + formalização ADR-037

## Metadados
- id: ISSUE-025
- tipo: CODE
- titulo: Análise integral do código, execução direta de testes e atualização de benchmark/documentação
- criado_em: 2026-02-17
- owner: Codex
- status: DONE

## 1. Objetivo
Executar uma auditoria operacional completa no repositório `pyfolds`, confirmar erros/falhas de execução, atualizar os artefatos de benchmark com resultados atuais e formalizar a decisão em ADR-037.

## 2. Escopo

### 2.1 Inclui:
- compilação completa de `src/`;
- execução da suíte principal de testes;
- atualização de `docs/assets/benchmarks_results.json`;
- regeneração de `docs/assets/BENCHMARKS.md`;
- criação do relatório executivo da execução (`EXEC-025`);
- criação e indexação da ADR-037;
- sincronização de governança (CSV + HUB).

### 2.2 Exclui:
- alterações de arquitetura de produto fora da análise operacional;
- refatoração ampla sem evidência de falha atual;
- inclusão de dependências novas apenas para benchmark.

## 3. Artefatos Gerados
- `docs/development/prompts/relatorios/ISSUE-025-analise-integral-benchmark-adr37.md`
- `docs/development/prompts/execucoes/EXEC-025-analise-integral-benchmark-adr37.md`
- `docs/governance/adr/ADR-037-analise-integral-issue-025-benchmark-refresh.md`
- `docs/governance/adr/INDEX.md`
- `docs/assets/benchmarks_results.json`
- `docs/assets/BENCHMARKS.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`

## 4. Riscos
- Risco: variabilidade de benchmark por ruído de ambiente (CPU compartilhada).
  Mitigação: uso de mediana com múltiplas amostras e seed fixa.
- Risco: warnings não-bloqueantes mascararem futuros problemas.
  Mitigação: registrar warnings explicitamente no EXEC para acompanhamento.

## 5. Critérios de Aceite
- ISSUE em conformidade com `tools/validate_issue_format.py`
- EXEC com passos executados e validações registradas
- Registro no `execution_queue.csv`
- `python tools/sync_hub.py` executado
- `HUB_CONTROLE.md` alterado no mesmo commit

## 6. PROMPT:EXECUTAR
```yaml
issue_id: "ISSUE-025"
tipo: "CODE"
titulo: "Análise integral do pyfolds e refresh de benchmark"

passos_obrigatorios:
  - "Compilar src com python -m compileall src"
  - "Executar testes com PYTHONPATH=src pytest -q"
  - "Atualizar benchmark com scripts/run_benchmarks.py"
  - "Gerar docs/assets/BENCHMARKS.md"
  - "Criar ADR-037 e atualizar INDEX"
  - "Criar EXEC-025 correspondente"
  - "Registrar ISSUE no execution_queue.csv"
  - "Rodar python tools/sync_hub.py"

validacao:
  - "python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-025-analise-integral-benchmark-adr37.md"
  - "python tools/sync_hub.py --check"
  - "python tools/check_issue_links.py docs/development/prompts/relatorios"
```
