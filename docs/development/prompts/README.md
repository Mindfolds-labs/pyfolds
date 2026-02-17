# 📁 Portal de Prompts Operacionais

Este diretório define como criar, analisar e executar issues com rastreabilidade no CSV/HUB.

---

## 🔢 Regra obrigatória de numeração (IA)

**Antes de criar qualquer nova ISSUE, a IA deve ler `docs/development/execution_queue.csv` e calcular o próximo número sequencial disponível no formato `ISSUE-NNN`.**

### Algoritmo obrigatório
1. Ler todas as linhas do CSV.
2. Extrair IDs no padrão `ISSUE-\d{3}` (ignorar `ISSUE-XXX-ESPECIAL`).
3. Calcular `max(NNN) + 1`.
4. Criar sempre no formato:
   - relatório: `docs/development/prompts/relatorios/ISSUE-[NNN]-[slug].md`
   - execução: `docs/development/prompts/execucoes/EXEC-[NNN]-[slug].md`
5. Registrar a nova ISSUE no `execution_queue.csv` com o mesmo número.
6. Sincronizar o HUB (`python tools/sync_hub.py`).

> Exemplo prático: se o maior ID for `ISSUE-016`, a próxima deve ser `ISSUE-017`.

---

## 🔄 Ciclo oficial
1. **CRIAR** (humano ou IA)
2. **ANALISAR** (humano)
3. **EXECUTAR** (Codex)
4. **FINALIZAR** (humano)

---

## 🧾 Entrega mínima obrigatória por ISSUE
- Relatório em `relatorios/ISSUE-[NNN]-[slug].md` (com análise e diagnóstico)
- Plano de execução em `execucoes/EXEC-[NNN]-[slug].md`
- Linha correspondente no `execution_queue.csv`
- HUB sincronizado

Sem esses 4 itens a issue não é considerada entregue.

---

## ✅ Checklist rápido de execução (IA)
1. Descobrir próximo número pelo CSV.
2. Criar relatório no padrão de auditoria/consolidação.
3. Criar arquivo de execução (`EXEC-[NNN]-...`).
4. Atualizar `execution_queue.csv`.
5. Rodar:
   - `python tools/sync_hub.py`
   - `python tools/sync_hub.py --check`
   - `python tools/check_issue_links.py docs/development/prompts/relatorios`

---

## 🔗 Referências
- [Relatórios](./relatorios/README.md)
- [Guia de formato](../guides/ISSUE-FORMAT-GUIDE.md)
- [Fila de execução](../execution_queue.csv)
- [HUB de controle](../HUB_CONTROLE.md)
