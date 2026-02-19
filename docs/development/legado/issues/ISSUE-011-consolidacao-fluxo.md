# ISSUE-011: Consolidação de Fluxo e Correção de Cards/Links

## Metadados

| Campo | Valor |
|---|---|
| Data | 2026-02-17 |
| Autor | Codex |
| Tipo | GOVERNANCE + DOCUMENTATION |
| Prioridade | Crítica |

## 1. Objetivo
Consolidar o fluxo operacional de issues (CRIAR → ANALISAR → EXECUTAR → FINALIZAR), corrigindo cards/links no HUB e alinhando status da fila de execução.

## 2. Escopo

### 2.1 Inclui:
- Revisão dos status das issues 001-010-ESPECIAL.
- Correção de cards e consistência do HUB.
- Correção de links e índice em `docs/development/prompts/README.md`.
- Registro da ISSUE-011 no CSV e log de execução.
- Execução de validações técnicas (links/hub/formato/compileall).

### 2.2 Exclui:
- Mudanças de código-fonte em `src/`.
- Mudanças de testes fora da validação prevista.
- Reestruturação radical da documentação.

## 3. Artefatos Gerados
- `docs/development/prompts/relatorios/ISSUE-011-consolidacao-fluxo.md`
- `docs/development/prompts/relatorios/ISSUE-011-ESPECIAL-consolidacao-fluxo.md`
- `docs/development/prompts/logs/ISSUE-011-ESPECIAL-consolidacao-fluxo-LOG.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`
- `docs/development/prompts/README.md`

## 4. Riscos
- Quebra de links durante correção de caminhos.
- Divergência HUB↔CSV após edições manuais.
- Inconsistência de status sem evidência histórica explícita.

## 5. Critérios de Aceite
- HUB com cards consistentes para ISSUE-001..ISSUE-011.
- Índice de relatórios no README de prompts.
- `python tools/check_links.py docs/ README.md` sem erros críticos.
- `python tools/sync_hub.py --check` sem divergências.
- `python -m compileall src/` sem `SyntaxError`.

## 6. PROMPT:EXECUTAR
```yaml
fase: CONSOLIDACAO_FLUXO_COMPLETO
prioridade: CRITICA
responsavel: CODEX

acoes:
  - validar_status_csv
  - ajustar_status_issues
  - corrigir_cards_hub
  - adicionar_indice_relatorios
  - validar_links
  - validar_sync_hub
  - validar_formato_issue
  - validar_compileall
  - registrar_log
```
