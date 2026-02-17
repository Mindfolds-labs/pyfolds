# Guia de Revisão — Engenharia de Interação (UX) para Documentação + Conformidade IEEE

## Objetivo
Padronizar revisões de documentação técnica para que sejam:
- fáceis de navegar (UX de documentação);
- rastreáveis (governança de mudanças);
- auditáveis sob referências IEEE/ISO já adotadas no projeto.

## Escopo
Aplicável a artefatos em `docs/development/` e `docs/governance/` que representem processo, fila de execução, ADRs e relatórios de issue.

## Checklist de UX da Documentação

### 1) Descoberta e navegação
- [ ] O documento informa claramente **Objetivo**, **Escopo** e **Público-alvo** no topo.
- [ ] Há links diretos para artefatos relacionados (HUB, fila CSV, ADR index, relatório da issue).
- [ ] O título e slug do arquivo seguem convenção (`ISSUE-XXX-*`, `ADR-XXX-*`).

### 2) Legibilidade e fluxo de leitura
- [ ] Seções seguem uma ordem previsível (metadados → objetivo → escopo → execução → achados → evidências).
- [ ] Achados estão priorizados por severidade (P0/P1/P2) e com comandos reprodutíveis.
- [ ] O leitor consegue entender “o que fazer agora” sem contexto externo.

### 3) Ação e fechamento
- [ ] Existem critérios de aceite objetivos.
- [ ] Próximas ações estão explícitas (issue/PR/ADR impactada).
- [ ] Há vínculo com logs/evidências para auditoria.

## Checklist de Conformidade IEEE/ISO (operacional)

### IEEE 828 (Configuração)
- [ ] Mudança registrada na fila (`execution_queue.csv`) com artefatos impactados.
- [ ] HUB sincronizado após atualização da fila (`tools/sync_hub.py`).

### IEEE 730 (Qualidade)
- [ ] Evidências de validação executadas e anexadas (docs checks, testes, validações).
- [ ] Não conformidades estão descritas com causa provável e recomendação.

### ISO/IEC 12207 (Processo)
- [ ] Resultado da atividade tem entrada/saída definidas (prompt + artefatos gerados).
- [ ] Há rastreabilidade entre decisão arquitetural (ADR), execução (issue) e evidência (logs).

## Métricas sugeridas para revisão rápida
- Tempo para encontrar evidência de um achado crítico (meta: < 2 minutos).
- Percentual de issues com critérios de aceite verificáveis (meta: 100%).
- Percentual de issues com vínculo ADR quando aplicável (meta: 100%).

## Comandos de validação recomendados
```bash
python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-*.md
python tools/check_issue_links.py docs/development/prompts/relatorios
python tools/check_links.py
python tools/sync_hub.py --check
```
