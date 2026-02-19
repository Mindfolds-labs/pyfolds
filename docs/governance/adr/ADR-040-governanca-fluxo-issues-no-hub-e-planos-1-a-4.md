# ADR-040 — Governança de execução: issues no HUB e plano ordenado 1→4

## Status
Aceito

## Contexto
Na auditoria funcional anterior, parte dos artefatos foi criada fora do fluxo oficial de prompts/HUB, dificultando rastreabilidade operacional e aderência ao padrão de `ISSUE-*` com frontmatter + seções obrigatórias.

Também foi solicitado um plano completo, em ordem de execução, para correção transversal de documentação, governança e pontos de código.

## Decisão
1. Registrar novos itens de execução no fluxo canônico:
   - `docs/development/prompts/relatorios/ISSUE-003..006-*.md`
   - `docs/development/prompts/execucoes/EXEC-003..006-*.md`
   - `docs/development/execution_queue.csv` + sincronização via `tools/sync_hub.py`.
2. Definir a ordem obrigatória de execução em **4 planos**:
   - **Plano 1:** reposicionamento das issues no HUB;
   - **Plano 2:** saneamento de links/documentação;
   - **Plano 3:** execução de validações e registro de evidências;
   - **Plano 4:** consolidação de correções de código/testes/docs.
3. Toda nova issue do ciclo deve passar por `tools/validate_issue_format.py` antes de ser considerada pronta para execução.

## Alternativas consideradas
1. Manter issues somente em `docs/governance/quality/issues/` sem HUB.
   - Prós: menor esforço imediato.
   - Contras: quebra do fluxo operacional e baixa visibilidade de fila ativa.

2. Migrar para fluxo canônico do HUB (decisão adotada).
   - Prós: rastreabilidade, padronização e execução ordenada por prioridade.
   - Contras: custo inicial de reestruturação documental.

## Consequências
- O planejamento volta a ser observável no HUB e na fila ativa.
- A execução ganha sequência explícita e auditável (1→4).
- Reduz risco de reincidência de artefatos fora do padrão.
