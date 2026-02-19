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
O HUB de controle concentrou conteúdo ativo e histórico, causando ruído operacional e dificultando navegação da fila atual.

## Análise Técnica
1. Manter somente cards de issues ativas no HUB principal.
2. Preservar histórico no diretório legado.
3. Criar split view dedicado para acesso rápido a relatórios e execuções.

## Requisitos Funcionais
- [x] RF-01: HUB principal com foco na fila ativa.
- [x] RF-02: Página split view publicada com atalhos operacionais.
- [x] RF-03: Links entre HUB, fila e prompts íntegros.

## Requisitos Não-Funcionais
- [x] RNF-01: Performance
- [x] RNF-02: Segurança

## Artefatos Esperados
- Atualização de `docs/development/HUB_CONTROLE.md`.
- Criação/ajuste de `docs/development/ISSUES_SPLIT_VIEW.md`.
- Atualização da linha da issue no `execution_queue.csv`.

## Critérios de Aceite
- [x] Formato da issue aprovado por `tools/validate_issue_format.py`.
- [x] Referências cruzadas válidas e sem links quebrados.
- [x] HUB refletindo item na fila ativa após `tools/sync_hub.py`.

## Riscos e Mitigações
- Risco: divergência entre queue CSV e cards do HUB.
- Mitigação: execução obrigatória de `python tools/sync_hub.py --check` após sincronização.

## PROMPT:EXECUTAR
```yaml
objetivo: "Limpar o HUB removendo cards históricos (legado), melhorar visual dos cards ativos e criar página split para navegação rápida."
issue_id: "ISSUE-002"
prioridade: "Alta"
area: "Governança/UX Docs"
```

## Rastreabilidade (IEEE 830)
| Requisito | Evidência |
| --- | --- |
| RF-01 | HUB principal reorganizado para fila ativa |
| RF-02 | Split view criada e vinculada no HUB |
| RF-03 | Validações de links e sync_hub executadas |
