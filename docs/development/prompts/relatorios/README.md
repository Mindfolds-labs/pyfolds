# 🧾 README — Relatórios de Issues

Os relatórios em `relatorios/` seguem padrão de **auditoria/consolidação** (ex.: ISSUE-003), com análise explícita e evidências.

---

## 📌 Formato recomendado (canônico atual)
1. Título do relatório (`RELATÓRIO ... ISSUE-NNN`)
2. Metadados em tabela (data, responsável, issue, tipo, status, normas)
3. **1. Sumário Executivo**
4. **2. Diagnóstico e Análise**
5. **3. Artefatos Atualizados**
6. **4. Execução Técnica**
7. **5. Riscos, Restrições e Mitigações**
8. **6. Critérios de Aceite e Status**

---

## 🔢 Regra de numeração obrigatória
Sempre usar o próximo `ISSUE-NNN` calculado a partir de `docs/development/execution_queue.csv`.

Se o maior for `ISSUE-016`, a próxima criação obrigatória é `ISSUE-017`.

---

## 📦 Entrega completa da ISSUE
Para cada relatório criado, também deve existir:
- `docs/development/prompts/execucoes/EXEC-[NNN]-[slug].md`
- linha correspondente no `execution_queue.csv`
- HUB sincronizado por `tools/sync_hub.py`

---

## ✅ Validação operacional
```bash
python tools/sync_hub.py
python tools/sync_hub.py --check
python tools/check_issue_links.py docs/development/prompts/relatorios
```
