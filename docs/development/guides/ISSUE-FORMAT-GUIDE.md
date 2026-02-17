# Guia de Implementação — Formato de ISSUE (Humano → IA)

## Objetivo
Padronizar o processo para que a IA execute corretamente: **número certo**, **formato certo** e **entrega completa**.

---

## 1) Regra de numeração (obrigatória)
A IA deve calcular o próximo ID com base no CSV oficial:
- fonte: `docs/development/execution_queue.csv`
- regra: `próximo = max(ISSUE-\d{3}) + 1`
- ignorar sufixos especiais (`-ESPECIAL`)

Exemplo: se o maior regular for `ISSUE-016`, a próxima é `ISSUE-017`.

---

## 2) Formato do relatório (seguir ISSUE-003)
Todo `ISSUE-[NNN]-[slug].md` deve usar padrão de auditoria/consolidação:

1. Título
2. Subtítulo
3. Tabela de metadados
4. `1. Sumário Executivo`
5. `2. Diagnóstico e Análise`
6. `3. Artefatos Atualizados`
7. `4. Execução Técnica`
8. `5. Riscos, Restrições e Mitigações`
9. `6. Critérios de Aceite e Status`

---

## 3) Arquivos obrigatórios por issue
1. `docs/development/prompts/relatorios/ISSUE-[NNN]-[slug].md`
2. `docs/development/prompts/execucoes/EXEC-[NNN]-[slug].md`
3. atualização em `docs/development/execution_queue.csv`
4. sincronização de `docs/development/HUB_CONTROLE.md`

Sem os 4 itens, não há entrega completa.

---

## 4) Modelo mínimo do EXEC
`EXEC-[NNN]-[slug].md` deve conter:
- tarefa
- contexto
- passos de execução
- validações
- atualização final de CSV/HUB

---

## 5) Checklist operacional
```bash
python tools/sync_hub.py
python tools/sync_hub.py --check
python tools/check_issue_links.py docs/development/prompts/relatorios
```
