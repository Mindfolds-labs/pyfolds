# ADR-009 — Governança operacional e checklist de release

## Status
Accepted

## Contexto
Além do formato em si, a confiabilidade depende de processo: runbook, observabilidade e critérios formais de liberação.

## Decisão
Formalizar governança de operação:

- runbook para falhas de leitura/escrita;
- telemetria/alertas para integridade de checkpoints;
- checklist pré-release com validações de serialização e treinamento.

## Consequências
### Positivas
- Melhor prontidão para incidentes.
- Releases mais previsíveis e auditáveis.

### Trade-offs
- Overhead processual adicional para cada release.
