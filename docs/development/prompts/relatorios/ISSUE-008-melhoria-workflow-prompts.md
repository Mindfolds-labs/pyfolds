# ISSUE-008 — Melhorar workflow de prompts (Criar → Analisar → Executar → Finalizar)

> **PMBOK Fase:** Iniciação + Planejamento  
> **Status:** ⏳ Planejada  
> **Área:** Documentação/Processo  
> **Data:** 2026-02-16  
> **Responsável:** Neto (Humano) + Codex (Executor)

---

## 🎯 Objetivo
Deixar o portal de prompts mais simples para uso diário: o humano cria e analisa, o Codex executa, e a aprovação final acontece no PR.

---

## 📋 Escopo

### Inclui
- ✅ README curto no portal com prompts reutilizáveis (criar/analisar e executar/finalizar).
- ✅ README de `relatorios/` com padrão obrigatório e bloco de `PROMPT:EXECUTAR`.
- ✅ Registro da ISSUE-008 na fila (`execution_queue.csv`).
- ✅ Atualização do HUB com referência da ISSUE-008.

### Exclui
- ❌ Alterações de código de produto (`src/`).
- ❌ Mudança funcional em testes/serialização.

---

## 📊 Artefatos afetados
- `docs/development/prompts/README.md`
- `docs/development/prompts/relatorios/README.md`
- `docs/development/prompts/relatorios/ISSUE-008-melhoria-workflow-prompts.md`
- `docs/development/prompts/logs/ISSUE-008-melhoria-workflow-prompts-LOG.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`

---

## ⏰ Plano de execução
1. Ajustar README do portal para versão curta e operacional.
2. Garantir no README de relatórios a estrutura mínima obrigatória.
3. Registrar ISSUE-008 no CSV com link curto para relatório.
4. Sincronizar HUB e validar links.

---

## 🚨 Riscos e mitigação
- [ ] Risco: excesso de texto no README | Mitigação: manter versão curta e com blocos objetivos.
- [ ] Risco: links quebrados após ajustes | Mitigação: rodar `python tools/validate_docs_links.py`.
- [ ] Risco: HUB e CSV divergirem | Mitigação: rodar `python tools/sync_hub.py --check`.

---

## ✅ Critérios de aceite
- [ ] Portal com ciclo claro: Criar, Analisar, Executar, Finalizar.
- [ ] Relatório com bloco reutilizável `PROMPT:EXECUTAR`.
- [ ] ISSUE-008 registrada na fila sem quebra de formato.
- [ ] HUB sincronizado com a fila.
- [ ] Validação de links e sync check verdes.

---

## 📝 PROMPT:EXECUTAR
<!-- PROMPT:EXECUTAR:INICIO -->
Você é o Codex atuando como Executor Técnico.

1) Leia este relatório e extraia objetivo, artefatos e critérios.
2) Atualize os READMEs do portal de prompts para versão curta, clara e reutilizável.
3) Registre ISSUE-008 em `docs/development/execution_queue.csv` usando como artefato principal o link do relatório.
4) Atualize `docs/development/HUB_CONTROLE.md` para incluir card da ISSUE-008.
5) Rode validações:
   - python tools/validate_docs_links.py
   - python tools/sync_hub.py --check
6) Faça commit e prepare PR para revisão humana.
<!-- PROMPT:EXECUTAR:FIM -->
