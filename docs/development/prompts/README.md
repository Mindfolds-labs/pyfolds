# 📁 Portal de Prompts Operacionais

Guia para o fluxo **humano → IA** com rastreabilidade completa.

---

## 🎯 Objetivo
Garantir que toda solicitação tenha:
1. relatório (`ISSUE-NNN`),
2. execução (`EXEC-NNN`),
3. registro no CSV,
4. HUB sincronizado.

---

## 🔄 Fluxo oficial (humano + IA)
1. **CRIAR (humano)**
   - descreve problema, escopo e critérios.
2. **ANALISAR (humano)**
   - aprova/reprova com checklist.
3. **EXECUTAR (IA)**
   - executa somente o escopo aprovado.
4. **FINALIZAR (humano)**
   - valida evidências e aprova PR.

---

## 🔢 Regra obrigatória de numeração (IA)
Antes de criar nova issue, a IA deve ler `docs/development/execution_queue.csv` e calcular o próximo `ISSUE-NNN` regular.

### Algoritmo
1. Extrair IDs `ISSUE-\d{3}`.
2. Ignorar variantes como `ISSUE-010-ESPECIAL`.
3. Calcular `max + 1`.
4. Criar:
   - `docs/development/prompts/relatorios/ISSUE-[NNN]-[slug].md`
   - `docs/development/prompts/execucoes/EXEC-[NNN]-[slug].md`
5. Registrar no CSV e sincronizar HUB.

> Exemplo: se o maior for `ISSUE-016`, a próxima obrigatória é `ISSUE-017`.

---

## 🧩 Prompt padrão para o HUMANO criar uma issue para IA

```markdown
CRIAR ISSUE PARA IA

TIPO: [CODE|DOCS|TEST|ADR|GOVERNANCE]
TITULO: [curto e objetivo]
JUSTIFICATIVA: [problema real]

INCLUI:
- item 1
- item 2

EXCLUI:
- item fora de escopo

ARTEFATOS:
- caminho/arquivo1
- caminho/arquivo2

RISCOS:
- risco | mitigação

Regras obrigatórias:
- descobrir próximo ISSUE-NNN pelo execution_queue.csv
- criar ISSUE-[NNN]-[slug].md e EXEC-[NNN]-[slug].md
- registrar no execution_queue.csv
- sincronizar HUB
```

---

## ✅ Prompt padrão para ANALISAR (humano)

```markdown
ANÁLISE DA ISSUE

Checklist:
- [ ] formato do relatório segue padrão ISSUE-003
- [ ] escopo inclui/exclui está claro
- [ ] artefatos estão explícitos
- [ ] riscos e mitigação definidos
- [ ] critérios de aceite verificáveis

Status:
- [ ] APROVADA para execução
- [ ] REPROVADA com ajustes
```

---

## 🚀 Prompt padrão para EXECUTAR (IA)

```markdown
Executar ISSUE-[NNN] conforme relatório aprovado.

Passos:
1) Aplicar apenas o escopo definido.
2) Atualizar os artefatos listados.
3) Criar/atualizar EXEC-[NNN].
4) Atualizar execution_queue.csv.
5) Rodar validações:
   - python tools/sync_hub.py
   - python tools/sync_hub.py --check
   - python tools/check_issue_links.py docs/development/prompts/relatorios
6) Commit + PR.
```
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
- [execution_queue.csv](../execution_queue.csv)
- [HUB_CONTROLE.md](../HUB_CONTROLE.md)
