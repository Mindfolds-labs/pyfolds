---
id: "ISSUE-008"
titulo: "Resiliência de testes sem torch (analisar + executar)"
prioridade: "Alta"
area: "Testes/Infra"
responsavel: "Codex"
criado_em: "2026-02-19"
normas:
  - ABNT NBR ISO/IEC 25010
  - IEEE 830
---
# ISSUE-008: Resiliência de testes sem torch (analisar + executar)

## Objetivo
Executar análise e implementação de preflight/skip para ambientes sem torch, mantendo qualidade do fluxo de validação.

## Contexto Técnico
Demanda derivada da auditoria funcional com necessidade de formalização no fluxo canônico do HUB.

## Análise Técnica
1. Confirmar escopo técnico e risco.
2. Aplicar mudanças controladas e validáveis.
3. Registrar evidências no HUB e relatórios.

## Requisitos Funcionais
- [ ] RF-01: escopo executado com rastreabilidade.
- [ ] RF-02: validações aplicáveis executadas.
- [ ] RF-03: evidências anexadas em relatório/execução.

## Requisitos Não-Funcionais
- [ ] RNF-01: Performance
- [ ] RNF-02: Segurança

## Artefatos Esperados
- Relatório ISSUE no padrão ABNT/IEEE.
- EXEC associado.
- Atualização de status no HUB.

## Critérios de Aceite
- [ ] `tools/validate_issue_format.py` aprovando o relatório.
- [ ] `tools/sync_hub.py --check` sem divergência.
- [ ] Links locais válidos.

## Riscos e Mitigações
- Risco: pendências cruzadas em docs e core.
- Mitigação: execução faseada e validação por checkpoints.

## PROMPT:EXECUTAR
```yaml
objetivo: "Executar análise e implementação de preflight/skip para ambientes sem torch, mantendo qualidade do fluxo de validação."
issue_id: "ISSUE-008"
prioridade: "Alta"
area: "Testes/Infra"
```

## Rastreabilidade (IEEE 830)
| Requisito | Evidência |
| --- | --- |
| RF-01 | Atualização de artefatos do issue |
| RF-02 | Logs de validação anexados |
| RF-03 | HUB e fila sincronizados |
