# Workflow Integrado — Da ISSUE à Implementação

Este documento conecta o fluxo documental (`CRIAR → ANALISAR → EXECUTAR → FINALIZAR`) com a execução real em código, testes e governança.

## 1. Objetivo
Garantir que toda mudança passe por:
- planejamento rastreável (issue),
- execução controlada (escopo e validações),
- fechamento auditável (log, fila e hub).

## 2. Fluxo integrado

### 2.0 Política de fases

O workflow opera em três fases de governança documental:

1. **Fase ativa**
   - Aceita abertura de novas `ISSUE-*` para planejamento e execução.
   - Fluxo completo `CRIAR → ANALISAR → EXECUTAR → FINALIZAR` habilitado.

2. **Fase freeze**
   - Novas `ISSUE-*` ficam restritas a correções críticas (ex.: regressão severa, segurança, bloqueio de release).
   - Demandas não críticas devem ser adiadas para a próxima fase ativa.

3. **Fase legado**
   - `ISSUE-*` existentes permanecem como consulta histórica.
   - Não há abertura de novas issues; priorizar manutenção mínima e rastreabilidade de encerramento.

A fase vigente deve estar explícita nos templates e checklists de validação.

### 2.1 CRIAR (Humano)
Entradas mínimas:
- tipo da issue (`CODE`, `DOCS`, `TEST`, `ADR`, `GOVERNANCE`),
- fase vigente (`ativa`, `freeze` ou `legado`),
- justificativa,
- escopo inclui/exclui,
- artefatos concretos,
- riscos + mitigação.

Saídas:
- arquivo `docs/development/prompts/relatorios/ISSUE-[N]-[slug].md`,
- linha em `docs/development/execution_queue.csv`.

### 2.2 ANALISAR (Humano)
Aprovar apenas quando houver:
- objetivo claro,
- escopo executável,
- critérios de aceite verificáveis,
- `PROMPT:EXECUTAR` completo.

Saída:
- autorização explícita para execução.

### 2.3 EXECUTAR (Codex)
Execução limitada aos artefatos listados no relatório.

Validações típicas:
- `python -m compileall src/` (se alteração em código),
- `python tools/check_links.py docs/ README.md` (se alteração em docs),
- `PYTHONPATH=src pytest tests/ -v` (se alteração em comportamento),
- `python tools/sync_hub.py --check` (consistência de governança).

Saídas:
- arquivos alterados conforme escopo,
- log em `docs/development/prompts/logs/ISSUE-[N]-[slug]-LOG.md`,
- commit e PR.

### 2.4 FINALIZAR (Humano)
Validar evidências, aprovar PR e concluir rastreabilidade.

Saídas:
- merge,
- status final na fila,
- HUB sincronizado.

## 3. Mapeamento com artefatos reais

```text
docs/development/prompts/relatorios/ISSUE-[N]-[slug].md
        ↓
src/pyfolds/*        docs/*        tests/*        docs/governance/adr/*
        ↓
docs/development/prompts/logs/ISSUE-[N]-[slug]-LOG.md
        ↓
docs/development/execution_queue.csv
        ↓
docs/development/HUB_CONTROLE.md
```

## 4. Regras operacionais
1. Não executar sem análise humana aprovada.
2. Não expandir escopo além dos artefatos listados.
3. Toda execução precisa de log e atualização da fila.
4. Mudanças de arquitetura exigem ADR e atualização de índice.
5. Mudanças estruturais só podem avançar com referência explícita de ADR aprovada.

## 5. Checklist rápido por issue
- [ ] Relatório criado com template canônico.
- [ ] Análise humana registrada.
- [ ] Execução com validações adequadas ao tipo.
- [ ] Log da execução atualizado.
- [ ] `execution_queue.csv` atualizado.
- [ ] `python tools/sync_hub.py --check` sem divergências.
