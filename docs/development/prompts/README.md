# 📁 Portal de Prompts Operacionais

Guia oficial do fluxo **humano → IA** para criação, execução e rastreabilidade de ISSUEs.

## 🎯 Objetivo
Garantir que toda execução tenha:
1. `ISSUE-NNN` (relatório de solicitação),
2. `EXEC-NNN` (execução técnica),
3. registro em `docs/development/execution_queue.csv`,
4. sincronização de `docs/development/HUB_CONTROLE.md`.

## 🔄 Fluxo oficial (humano + IA)
1. **CRIAR (humano):** descreve problema, escopo e critérios.
2. **ANALISAR (humano):** aprova/reprova com checklist.
3. **EXECUTAR (IA):** implementa o escopo aprovado.
4. **FINALIZAR (humano):** valida evidências e aprova PR.

## 🔢 Regra obrigatória de numeração (IA)
Antes de criar uma nova ISSUE, a IA deve ler `docs/development/execution_queue.csv` e calcular o próximo `ISSUE-NNN` regular.

### Algoritmo obrigatório
1. Ler todas as linhas do CSV.
2. Extrair IDs no padrão `ISSUE-\d{3}`.
3. Ignorar variantes como `ISSUE-010-ESPECIAL`.
4. Calcular `max(NNN) + 1`.
5. Criar os dois artefatos com o mesmo número:
   - `docs/development/prompts/relatorios/ISSUE-[NNN]-[slug].md`
   - `docs/development/prompts/execucoes/EXEC-[NNN]-[slug].md`
6. Registrar a ISSUE no `execution_queue.csv`.

> Exemplo: se o maior ID regular é `ISSUE-017`, o próximo obrigatório é `ISSUE-018`.

## 🧩 Estrutura de documentação (sem conflito de formato)
Há **dois padrões complementares** no diretório:

- **Padrão de ISSUE para validação automática** (`tools/validate_issue_format.py`):
  - obrigatório para arquivos `ISSUE-[NNN]-*.md` novos;
  - requer seções `Metadados`, `Objetivo`, `Escopo`, `Artefatos`, `Riscos`, `Critérios` e `PROMPT:EXECUTAR` em YAML.
- **Padrão canônico de relatório técnico final** (`ISSUE-003-auditoria-completa.md`):
  - referência para corpo analítico e governança de entrega;
  - deve ser espelhado na seção de relatório técnico dentro das novas ISSUEs.

## ✅ Fluxo obrigatório de execução (IA)
**A execução só é válida quando os passos abaixo ocorrem no mesmo commit de entrega:**

1. Descobrir próximo `ISSUE-NNN` no `execution_queue.csv`.
2. Criar/atualizar `ISSUE-[NNN]-[slug].md`.
3. Criar/atualizar `EXEC-[NNN]-[slug].md`.
4. Atualizar `docs/development/execution_queue.csv` com a mesma ISSUE.
5. Executar `python tools/sync_hub.py`.
6. Confirmar que `docs/development/HUB_CONTROLE.md` foi alterado.
7. Validar consistência com:
   - `python tools/sync_hub.py --check`
   - `python tools/check_issue_links.py docs/development/prompts/relatorios`

> Se `execution_queue.csv` mudar e `HUB_CONTROLE.md` não mudar no commit, a entrega está incompleta.

## ✅ Prompt padrão para ANALISAR (humano)
```markdown
ANÁLISE DA ISSUE

Checklist:
- [ ] formato da ISSUE passa no validador
- [ ] seção de relatório técnico segue referência ISSUE-003
- [ ] escopo inclui/exclui está claro
- [ ] artefatos estão explícitos
- [ ] riscos e mitigação definidos
- [ ] critérios de aceite verificáveis

Status:
- [ ] APROVADA para execução
- [ ] REPROVADA com ajustes
```

## 🚀 Prompt padrão para EXECUTAR (IA)
```markdown
Executar ISSUE-[NNN] conforme relatório aprovado.

Passos:
1) Aplicar apenas o escopo definido.
2) Atualizar os artefatos listados.
3) Criar/atualizar EXEC-[NNN].
4) Atualizar execution_queue.csv.
5) Rodar python tools/sync_hub.py.
6) Garantir alteração de HUB_CONTROLE.md no mesmo commit.
7) Rodar validações:
   - python tools/sync_hub.py --check
   - python tools/check_issue_links.py docs/development/prompts/relatorios
8) Commit + PR.
```

## 🔗 Referências
- [Relatórios](./relatorios/README.md)
- [Modelo de ISSUE](./relatorios/ISSUE-000-template.md)
- [execution_queue.csv](../execution_queue.csv)
- [HUB_CONTROLE.md](../HUB_CONTROLE.md)
