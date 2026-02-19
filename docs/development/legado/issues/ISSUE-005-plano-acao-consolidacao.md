# ISSUE-005 — Plano de Ação de Consolidação Total

## Contexto
Este plano operacionaliza os achados da auditoria completa registrada em `ISSUE-003-auditoria-completa.md`, com foco em conformidade com IEEE 828, IEEE 730 e ISO/IEC 12207.

## Objetivo
Fechar gaps críticos e médios identificados na auditoria, elevando o projeto de maturidade **3 (Definido)** para um estado auditável com quality gates documentais e rastreabilidade de processo.

## Escopo
- Documentação de processo e governança.
- API pública e rastreabilidade ADR↔código.
- Automação de validação documental no CI.
- Atualização de exemplos e padronização de estruturas.

## Backlog Prioritário

### Sprint 1 — Fundação (3 dias)
1. Criar `CONTRIBUTING.md` canônico na raiz com ponte para `docs/development/CONTRIBUTING.md`.
2. Criar `CHANGELOG.md` inicial versionado a partir da `2.0.0`.
3. Preencher `docs/development/release_process.md` com fluxo completo e checklist.
4. Corrigir portal `docs/README.md` para entrypoints existentes.

### Sprint 2 — Qualidade (3 dias)
1. Criar workflow para validação de docstrings públicas em PR.
2. Criar workflow para validação de links quebrados em docs.
3. Definir convenção de referência ADR nos módulos `core/` e `serialization/`.
4. Revisar exemplos críticos para aderência à API atual.

### Sprint 3 — Governança e Automação (2 dias)
1. Padronizar estrutura de testes (`performance` vs `perf`) com decisão explícita.
2. Adicionar templates de Issue e PR com checklist de conformidade.
3. Criar verificador automatizado de rastreabilidade ADR↔código.

## Critérios de Aceite da ISSUE-005
- Todos os gaps críticos (C01–C04) da auditoria resolvidos.
- CI com validação de docstrings e links habilitado.
- HUB e fila sincronizados sem divergência.
- Evidências de rastreabilidade ADR↔implementação para módulos críticos.

## Dependências
- Aprovação de governança para os workflows novos.
- Alinhamento de mantenedores sobre convenção ADR em código.

## Evidências esperadas
- Commits/documentos nas trilhas listadas em `execution_queue.csv`.
- Execução verde de `python tools/sync_hub.py --check`.
- Pull request de consolidação com checklist completo.
