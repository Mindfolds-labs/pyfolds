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


## 6. Governança do projeto `pyfolds-board` (GitHub Projects)

### 6.1 Criação do board
- Nome do projeto: `pyfolds-board`.
- Template base: **Automated Kanban**.
- Escopo recomendado: projeto de organização (quando disponível) para consolidar issues/PRs de múltiplos repositórios.

Fluxo recomendado (CLI):
1. Criar o projeto com nome `pyfolds-board`.
2. Aplicar o template `Automated Kanban`.
3. Confirmar os campos automáticos `Status`, `Assignees`, `Labels`, `Repository`, `Linked pull requests`.

> Se o template não for aplicável via CLI no ambiente atual, criar o projeto vazio com o mesmo nome e replicar manualmente as colunas e automações descritas nas seções 6.2 a 6.5.

### 6.2 Colunas obrigatórias
O board deve manter exatamente o seguinte fluxo lógico de status:
- **Backlog/To-Do** (entrada padrão de novas demandas e incidentes);
- **In Progress** (execução ativa);
- **Review/Blocked** (em revisão técnica ou bloqueado);
- **Done** (encerrado, com critérios de saída atendidos).

Validação operacional:
- não pode existir coluna intermediária fora dessa taxonomia sem ADR/decisão de governança;
- qualquer renomeação exige atualização deste documento e do HUB.

### 6.3 Automações de movimento por status de issue/PR
Configurar automações para manter rastreabilidade entre Issue/PR e coluna do board:

1. **Issue aberta**
   - Ação: adicionar item ao projeto em **Backlog/To-Do**.
2. **Issue marcada como `in progress`** (label) **ou vinculada a PR em aberto**
   - Ação: mover para **In Progress**.
3. **PR aberta para a issue**
   - Ação: mover para **Review/Blocked** quando entrar em etapa de revisão.
4. **PR convertida em draft** ou issue com label `blocked`
   - Ação: manter/retornar para **Review/Blocked**.
5. **PR mergeada + issue fechada**
   - Ação: mover para **Done** apenas se regra da seção 6.5 for satisfeita.

### 6.4 Incidentes entram direto no Backlog
Cards gerados por incidentes (ex.: via `tools/create_failure_issues.py`) devem ser adicionados diretamente em **Backlog/To-Do**.

Padrão já suportado no repositório:
- `tools/create_failure_issues.py --project-title pyfolds-board --target-column "Backlog/To-Do"`

Esse comando deve ser o default operacional para ingestão de incidentes no board.

### 6.5 Regra de Done (Definition of Done do board)
Um card só pode ir para **Done** quando **todos** os critérios abaixo forem verdadeiros:
1. execução técnica concluída (implementação + validações previstas no tipo de mudança);
2. evidências registradas em log de execução da issue;
3. HUB sincronizado com `python tools/sync_hub.py` (ou `--check` sem divergências após sync);
4. issue/PR em estado final coerente (`closed`/`merged`, quando aplicável).

Se qualquer critério falhar, o card deve permanecer em **Review/Blocked**.

### 6.6 Auditoria e cadência de revisão
- Revisão semanal do board para identificar cards órfãos (sem issue/sem PR/sem owner).
- Revisão por release para garantir que nenhum item foi movido para **Done** sem HUB sincronizado.
- Divergências devem abrir issue de governança (`type: GOVERNANCE`).
