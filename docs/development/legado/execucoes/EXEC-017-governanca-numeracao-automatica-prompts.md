# EXECUÇÃO — ISSUE-017
## Governança de numeração automática e entrega completa de prompts

## Tarefa
Garantir que toda nova ISSUE seja criada com o próximo número sequencial obtido do `execution_queue.csv`, com entrega completa de relatório e execução.

## Regra operacional obrigatória (IA)
1. Ler `docs/development/execution_queue.csv`.
2. Encontrar o maior `ISSUE-NNN` regular.
3. Criar a próxima issue como `ISSUE-(NNN+1)`.
4. Criar também `EXEC-(NNN+1)` correspondente.
5. Registrar no CSV e sincronizar HUB.

## Passos executados nesta ISSUE
- Atualizar `docs/development/prompts/README.md` com regra de numeração automática.
- Atualizar `docs/development/prompts/relatorios/README.md` com padrão de entrega completa.
- Atualizar `docs/development/guides/ISSUE-FORMAT-GUIDE.md` com algoritmo obrigatório.
- Criar `ISSUE-017` e `EXEC-017`.
- Atualizar status da fila para ISSUE-012 a ISSUE-015 como concluídas.

## Validações
- `python tools/sync_hub.py`
- `python tools/sync_hub.py --check`
- `python tools/check_issue_links.py docs/development/prompts/relatorios`
