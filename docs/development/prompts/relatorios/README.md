# 🧾 README — Relatórios de Issues

Cada relatório (`ISSUE-XXX-slug.md`) é a fonte única para execução da issue.

---

## ✅ Estrutura mínima obrigatória
- Objetivo
- Escopo (inclui/exclui)
- Artefatos afetados
- Critérios de aceite
- `PROMPT:EXECUTAR`

Template base:
- [`ISSUE-000-template.md`](./ISSUE-000-template.md)

---

## 🔄 Fluxo recomendado no próprio relatório
1. **CRIAR** — issue documentada e registrada no CSV.
2. **ANALISAR** — humano valida objetivo, escopo e critérios.
3. **EXECUTAR** — Codex executa conforme `PROMPT:EXECUTAR`.
4. **FINALIZAR** — humano aprova no PR.

---

## 🧩 Bloco pronto para copiar
```markdown
## 📝 PROMPT:EXECUTAR
<!-- PROMPT:EXECUTAR:INICIO -->
Você é o Codex atuando como Executor Técnico.

1) Leia este relatório e extraia objetivo, artefatos e critérios de aceite.
2) Execute somente os artefatos listados.
3) Valide:
   - python -m compileall src/
   - python tools/validate_docs_links.py
   - python tools/sync_hub.py --check
   - PYTHONPATH=src pytest tests/ -v
4) Atualize docs/development/execution_queue.csv.
5) Atualize o log em ../logs/ISSUE-XXX-slug-LOG.md.
6) Faça commit e deixe o PR ready for review.
<!-- PROMPT:EXECUTAR:FIM -->
```

---

## 📌 Convenção de nome
- `ISSUE-XXX-slug.md`

Exemplos:
- `ISSUE-003-auditoria-completa.md`
- `ISSUE-008-melhoria-workflow-prompts.md`
