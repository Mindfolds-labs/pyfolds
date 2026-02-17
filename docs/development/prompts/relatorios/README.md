# 🧾 README — Relatórios de Issues

Este diretório contém relatórios no padrão **auditoria/consolidação** (referência obrigatória: `ISSUE-003-auditoria-completa.md`).

---

## ✅ Estrutura canônica (seguir à risca)
Todo relatório novo deve conter, nesta ordem:

1. **Título principal** (ex.: `# RELATÓRIO DE CONSOLIDAÇÃO — ISSUE-017`)
2. **Subtítulo** (contexto da issue)
3. **Tabela de metadados** (Data, Responsável/Auditor, Issue, Tipo, Status, Normas)
4. `## 1. Sumário Executivo`
5. `## 2. Diagnóstico e Análise`
6. `## 3. Artefatos Atualizados`
7. `## 4. Execução Técnica`
8. `## 5. Riscos, Restrições e Mitigações`
9. `## 6. Critérios de Aceite e Status`

Se não tiver essa estrutura, a issue está incompleta.

---

## 🔢 Numeração obrigatória
- Descobrir próximo número no `docs/development/execution_queue.csv`.
- Criar `ISSUE-[NNN]-[slug].md` e `EXEC-[NNN]-[slug].md` com o mesmo NNN.

---

## 📦 Entrega obrigatória por issue
- Relatório em `relatorios/`
- Execução em `execucoes/`
- Linha no `execution_queue.csv`
- HUB sincronizado por `tools/sync_hub.py`

---

## ✅ Validações mínimas
```bash
python tools/sync_hub.py
python tools/sync_hub.py --check
python tools/check_issue_links.py docs/development/prompts/relatorios
```

---

## 📚 Referência de formato
- `ISSUE-003-auditoria-completa.md`
