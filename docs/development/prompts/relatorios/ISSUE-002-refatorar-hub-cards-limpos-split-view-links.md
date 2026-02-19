---
id: "ISSUE-002"
titulo: "Refatorar HUB com cards limpos e split view de links"
prioridade: "Alta"
area: "Governança/UX Docs"
responsavel: "Codex"
criado_em: "2026-02-19"
normas:
  - ABNT NBR ISO/IEC 25010
  - IEEE 830
---
# ISSUE-002: Refatorar HUB com cards limpos e split view de links

## Objetivo
Limpar o HUB removendo cards históricos (legado), melhorar visual dos cards ativos e criar página split para navegação rápida.

## Contexto Técnico
O HUB apresentava ruído visual e baixa separação entre itens ativos e legados, reduzindo legibilidade operacional.

## Análise Técnica
1. Padronizar cards em bloco canônico do HUB.
2. Isolar navegação rápida em split view.
3. Sincronizar fila via `execution_queue.csv` + `tools/sync_hub.py`.

## Requisitos Funcionais
- [x] RF-01: atualizar `docs/development/HUB_CONTROLE.md` com cards ativos limpos.
- [x] RF-02: manter visão rápida de navegação para equipe.
- [x] RF-03: manter sincronização com `execution_queue.csv`.

## Requisitos Não-Funcionais
- [x] RNF-01: Performance
- [x] RNF-02: Segurança

## Artefatos Esperados
- Atualizações em HUB/documentação de desenvolvimento.
- Fila e cards sincronizados com script oficial.

## Critérios de Aceite
- [x] Cards ativos renderizados corretamente no HUB.
- [x] Links internos válidos.
- [x] `tools/sync_hub.py --check` sem divergência.

## Riscos e Mitigações
- Risco: divergência entre CSV e blocos de cards.
- Mitigação: sincronização determinística obrigatória com `tools/sync_hub.py`.

## PROMPT:EXECUTAR
```yaml
objetivo: "Refatorar HUB com cards limpos e split view de links"
issue_id: "ISSUE-002"
prioridade: "Alta"
area: "Governança/UX Docs"
```

## Rastreabilidade (IEEE 830)
| Requisito | Evidência |
| --- | --- |
| RF-01 | HUB atualizado com cards e tabela resumida |
| RF-02 | Links rápidos no painel lateral |
| RF-03 | Fila sincronizada por `tools/sync_hub.py` |
