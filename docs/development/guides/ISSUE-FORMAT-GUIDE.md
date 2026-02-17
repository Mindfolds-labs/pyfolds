# Guia de Implementação — Formato Padrão de ISSUE para IA

## Objetivo
Padronizar a criação de ISSUE + EXECUÇÃO com numeração automática via CSV e relatório no formato de auditoria/consolidação.

---

## 1) Regra de numeração (obrigatória)
A IA **deve** descobrir o próximo ID pela fila oficial:
- Arquivo-fonte: `docs/development/execution_queue.csv`
- Regra: `próximo = max(ISSUE-\d{3}) + 1`
- Ignorar sufixos não sequenciais (ex.: `ISSUE-010-ESPECIAL`)

### Exemplo
Último regular: `ISSUE-016` → próxima criação obrigatória: `ISSUE-017`.

---

## 2) Artefatos obrigatórios por nova ISSUE
1. Relatório: `docs/development/prompts/relatorios/ISSUE-[NNN]-[slug].md`
2. Execução: `docs/development/prompts/execucoes/EXEC-[NNN]-[slug].md`
3. Registro no CSV: `docs/development/execution_queue.csv`
4. HUB sincronizado: `docs/development/HUB_CONTROLE.md` via `tools/sync_hub.py`

Sem esses quatro artefatos, a entrega está incompleta.

---

## 3) Estrutura do relatório (canônica)
Use o formato de auditoria/consolidação (referência: `ISSUE-003`):
- Metadados
- 1. Sumário Executivo
- 2. Diagnóstico e Análise
- 3. Artefatos Atualizados
- 4. Execução Técnica
- 5. Riscos, Restrições e Mitigações
- 6. Critérios de Aceite e Status

---

## 4) Estrutura mínima do arquivo EXEC
`EXEC-[NNN]-[slug].md` deve conter:
- Tarefa
- Contexto (issue + problema)
- Passos objetivos
- Validações
- Regra de atualização de status no CSV/HUB

---

## 5) Validações recomendadas
```bash
python tools/sync_hub.py
python tools/sync_hub.py --check
python tools/check_issue_links.py docs/development/prompts/relatorios
```
