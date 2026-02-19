# ISSUE-007 — Consolidação Final / Normalização Total

> **PMBOK Fase:** Iniciação → Planejamento  
> **Status:** ⏳ Planejada  
> **Área:** docs + governança + rastreabilidade  
> **Data:** 2026-02-16  
> **Responsável:** Codex (Executor) / Maintainers (Aprovação)

---

## 📌 Contexto
Há inconsistências de rastreabilidade no ciclo operacional (CRIAR → ANALISAR → EXECUTAR → FINALIZAR), com artefatos distribuídos em `prompts/` na raiz e links legados em documentos de governança.

## 🎯 Objetivo
Normalizar o fluxo de trabalho e os caminhos canônicos dos artefatos operacionais, consolidando o workflow no `README.md` e migrando o conteúdo de `prompts/` para `docs/development/prompts/`.

## 📋 Escopo

### Inclui
- Migração da pasta `prompts/` (raiz) para trilha canônica em `docs/development/prompts/`.
- Criação de portal oficial `docs/development/prompts/README.md`.
- Atualização de referências antigas para novos caminhos.
- Atualização da fila (`execution_queue.csv`) e sincronização do HUB.
- Execução de validações técnicas (sintaxe, links, HUB, testes).

### Exclui
- Mudanças arquiteturais profundas no core.
- Reescrita completa de documentação técnica fora dos pontos de referência.

## 📊 Artefatos Afetados
- `docs/development/prompts/README.md` (novo)
- `docs/development/prompts/relatorios/` (canônico)
- `docs/development/prompts/logs/` (canônico)
- `prompts/` (stub de compatibilidade)
- `README.md` (bloco Workflow v6)
- `docs/README.md`
- `docs/development/HUB_CONTROLE.md`
- `docs/development/execution_queue.csv`
- `tools/validate_docs_links.py` (sem hardcode legado)
- `tools/sync_hub.py` (verificado)

## ⏰ Plano de Execução
- Etapa 1 — Criar estrutura canônica e relatório/log da ISSUE-007.
- Etapa 2 — Migrar prompts da raiz para `docs/development/prompts/`.
- Etapa 3 — Atualizar workflow no README e referências legadas.
- Etapa 4 — Sincronizar fila/HUB e validar baseline técnico.

## 🚨 Riscos & Mitigação
- [ ] Links quebrados após migração | Mitigação: rodar validador de links e corrigir referências.
- [ ] Divergência HUB↔CSV | Mitigação: rodar `python tools/sync_hub.py --check` antes do commit.
- [ ] Dependência externa do path antigo | Mitigação: manter stub de compatibilidade em `prompts/README.md`.

## ✅ Critérios de Aceite
- [ ] Sem referências vivas para `docs/development/prompts/relatorios` (exceto stub de compatibilidade).
- [ ] `python tools/validate_docs_links.py` verde.
- [ ] `python tools/sync_hub.py --check` verde.
- [ ] `pytest tests/ -v` verde.
- [ ] ISSUE-007 registrada e rastreável em CSV + HUB + LOG.

## 📝 PROMPT:EXECUTAR
<!-- PROMPT:EXECUTAR:INICIO -->
1. Migrar `prompts/` para `docs/development/prompts/` preservando histórico via `git mv`.
2. Criar portal oficial em `docs/development/prompts/README.md`.
3. Atualizar `README.md` da raiz com workflow v6 e links canônicos.
4. Corrigir referências legadas para a trilha canônica em `docs/development/prompts/relatorios`.
5. Atualizar `execution_queue.csv` com ISSUE-007 e trilhas canônicas.
6. Sincronizar HUB com `python tools/sync_hub.py` e validar com `--check`.
7. Executar validações de sintaxe, links e testes.
8. Atualizar LOG da ISSUE-007 e abrir PR para revisão.
<!-- PROMPT:EXECUTAR:FIM -->
