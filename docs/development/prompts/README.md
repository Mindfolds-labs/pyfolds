# 📁 Portal de Prompts Operacionais

Guia curto para usar o ciclo de issues com aprovação humana no PR.

---

## 🔄 Ciclo oficial
1. **CRIAR** (humano)
2. **ANALISAR** (humano)
3. **EXECUTAR** (Codex)
4. **FINALIZAR** (humano)

> O detalhe completo fica dentro de cada relatório em `relatorios/ISSUE-XXX-slug.md`.

---

## 🗂️ Estrutura
- `relatorios/` → plano completo da issue + prompts
- `logs/` → evidência de execução

---

## 🧩 Prompt curto — CRIAR + ANALISAR
```markdown
Codex, criar ISSUE-[N] em docs/development/prompts/relatorios/ISSUE-[N]-[slug].md
e preparar para análise humana.

Inclua no relatório:
- objetivo
- escopo (inclui/exclui)
- artefatos
- riscos
- critérios de aceite
- bloco PROMPT:EXECUTAR

Depois:
1) registrar no docs/development/execution_queue.csv
2) rodar python tools/sync_hub.py
3) rodar python tools/sync_hub.py --check
```

---

## 🚀 Prompt curto — EXECUTAR
```markdown
Codex, executar ISSUE-[N] usando o relatório
docs/development/prompts/relatorios/ISSUE-[N]-[slug].md.

Siga o PROMPT:EXECUTAR do relatório e valide:
- python -m compileall src/
- python tools/validate_docs_links.py
- python tools/sync_hub.py --check
- PYTHONPATH=src pytest tests/ -v

Atualize:
- docs/development/execution_queue.csv
- docs/development/prompts/logs/ISSUE-[N]-[slug]-LOG.md

Finalize com commit e PR ready for review.
```

---

## ✅ Prompt curto — FINALIZAR (humano)
1. Revisar PR e evidências.
2. Aprovar ou solicitar ajuste.
3. Fazer merge quando estiver OK.

---

## 🔗 Links úteis
- [HUB_CONTROLE.md](../HUB_CONTROLE.md)
- [execution_queue.csv](../execution_queue.csv)
- [relatorios/](./relatorios/)
- [logs/](./logs/)
