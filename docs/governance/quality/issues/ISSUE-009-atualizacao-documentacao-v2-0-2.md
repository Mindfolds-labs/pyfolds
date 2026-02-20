# ISSUE-009 — Atualizar documentação de hardening da versão 2.0.2

- **Status:** Aberta
- **Tipo:** Governança / Documentação
- **Severidade:** Alta
- **Versão-alvo:** 2.0.2

## Contexto

A auditoria técnica da versão `2.0.2` validou avanços relevantes em segurança de checkpoints, telemetria e tolerância a falhas. Falta consolidar no acervo canônico a trilha de continuidade para verificação de integridade em runtime (VRAM) e alinhamento do HUB com os novos artefatos.

## Escopo

1. Consolidar ADR de monitoramento periódico de integridade de pesos.
2. Publicar relatório de execução de testes da entrega.
3. Sincronizar HUB de desenvolvimento com a nova issue concluída.
4. Garantir rastreabilidade no `execution_queue.csv`.

## Critérios de aceite

- ADR canônica adicionada ao índice oficial.
- Testes unitários novos executados com evidência em relatório.
- HUB atualizado com entrada da issue e links para execução/relatório.
- Changelog com menção ao novo monitor de integridade.

## Artefatos esperados

- `docs/governance/adr/ADR-046-monitoramento-periodico-de-integridade-dos-pesos.md`
- `docs/development/prompts/relatorios/ISSUE-014-hardening-integridade-pesos-vram.md`
- `docs/development/prompts/execucoes/EXEC-014-hardening-integridade-pesos-vram.md`
