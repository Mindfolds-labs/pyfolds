# HUB de Legado — Issues e Execuções

Este hub centraliza o acervo histórico da fase de legado.

## Objetivo

- manter rastreabilidade histórica sem perder referências antigas;
- orientar criação de novas issues com numeração automática;
- manter o modelo de relatório baseado na `ISSUE-003`.

## Referências legadas

- Issues históricas: [`issues/`](./issues/)
- Execuções históricas: [`execucoes/`](./execucoes/)
- Relatórios históricos: [`relatorios/`](./relatorios/)
- Modelo de referência: [`issues/ISSUE-003-auditoria-completa.md`](./issues/ISSUE-003-auditoria-completa.md)

## Regra de numeração

- Issue: iniciar em `ISSUE-001` quando ambiente vazio.
- ADR: iniciar em `ADR-0001` quando ambiente vazio.
- Com histórico existente: usar **sempre** o próximo número (`max + 1`).

A automação oficial está no utilitário `tools/id_registry.py`, consumido por `tools/create_issue_report.py --auto-id`.

## Registro no ambiente

A cada criação de issue via script, o último identificador é salvo em:

- `docs/development/legado/id_registry.json`

Esse registro evita perda de sequência operacional durante o fluxo humano → IA.
