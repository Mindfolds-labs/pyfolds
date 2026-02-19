---
id: "ISSUE-003"
titulo: "Plano 1 — Reposicionar issues no fluxo oficial do HUB"
prioridade: "Alta"
area: "Governança/Fluxo"
responsavel: "Codex"
criado_em: "2026-02-19"
normas:
  - ABNT NBR ISO/IEC 25010
  - IEEE 830
---
# ISSUE-003: Plano 1 — Reposicionar issues no fluxo oficial do HUB

## Objetivo
Reposicionar issues da auditoria para o fluxo oficial (Portal HUB + prompts/relatorios), garantindo rastreabilidade e padrão validável.

## Contexto Técnico
A auditoria funcional anterior criou artefatos fora do fluxo principal do HUB (`docs/development/HUB_CONTROLE.md`) e fora do padrão validável do diretório `docs/development/prompts/relatorios`.

## Análise Técnica
1. Alinhar artefatos ao padrão de governança ativo.
2. Preservar rastreabilidade do histórico já produzido.
3. Garantir sincronização entre CSV da fila, cards do HUB e documentos de execução.

## Requisitos Funcionais
- [ ] RF-01: artefatos criados no local canônico do fluxo de prompts.
- [ ] RF-02: vínculo explícito com execução (`EXEC-*`) e com HUB.
- [ ] RF-03: validações do fluxo executadas e registradas.

## Requisitos Não-Funcionais
- [ ] RNF-01: Performance
- [ ] RNF-02: Segurança

## Artefatos Esperados
- Documento de relatório de issue no padrão `ISSUE-XXX-*.md`.
- Documento de execução associado em `docs/development/prompts/execucoes/`.
- Linha correspondente em `docs/development/execution_queue.csv` e sincronização no HUB.

## Critérios de Aceite
- [ ] Formato da issue aprovado por `tools/validate_issue_format.py`.
- [ ] Referências cruzadas válidas e sem links quebrados.
- [ ] HUB refletindo item na fila ativa após `tools/sync_hub.py`.

## Riscos e Mitigações
- Risco: divergência entre queue CSV e cards do HUB.
- Mitigação: execução obrigatória de `python tools/sync_hub.py --check` após sincronização.

## PROMPT:EXECUTAR
```yaml
objetivo: "Reposicionar issues da auditoria para o fluxo oficial (Portal HUB + prompts/relatorios), garantindo rastreabilidade e padrão validável."
issue_id: "ISSUE-003"
prioridade: "Alta"
area: "Governança/Fluxo"
```

## Rastreabilidade (IEEE 830)
| Requisito | Evidência |
| --- | --- |
| RF-01 | Arquivo de issue criado no diretório canônico |
| RF-02 | EXEC associado e fila do HUB atualizada |
| RF-03 | Logs dos validadores anexados |
