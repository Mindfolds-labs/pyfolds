# 📁 Portal de Prompts Operacionais

Guia prático para executar o ciclo de issues com aprovação humana no PR.

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

Padronização de formato (ISSUE-009):
- `../templates/ISSUE-IA-TEMPLATE.md`
- `../guides/ISSUE-FORMAT-GUIDE.md`
- `../checklists/ISSUE-VALIDATION.md`

---

## 🆕 Como CRIAR uma boa ISSUE

Antes de pedir a criação da issue, preencha na ordem:

1. **TIPO:** `CODE`, `DOCS`, `TEST`, `ADR`, `GOVERNANCE`
2. **TÍTULO curto:** até 10 palavras
3. **JUSTIFICATIVA:** problema real que será resolvido
4. **ESCOPO (inclui/exclui):** limites claros
5. **ARTEFATOS:** lista explícita de arquivos/pastas
6. **RISCOS:** risco + mitigação

Prompt recomendado:

```markdown
CRIAR ISSUE

TIPO: [CODE|DOCS|TEST|ADR|GOVERNANCE]
TITULO: [curto e objetivo]
JUSTIFICATIVA: [uma frase]

INCLUI:
- item 1
- item 2

EXCLUI:
- item 1

ARTEFATOS:
- caminho/arquivo1
- caminho/arquivo2

RISCOS:
- risco 1 | mitigação

Criar em: docs/development/prompts/relatorios/ISSUE-[N]-[slug].md
```

Após criar:
1) registrar no `docs/development/execution_queue.csv`
2) rodar `python tools/sync_hub.py`
3) rodar `python tools/sync_hub.py --check`
4) rodar `python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-[N]-[slug].md`
5) rodar `python tools/check_issue_links.py docs/development/prompts/relatorios`

---

## ✅ Como ANALISAR uma ISSUE antes de executar

Checklist humano obrigatório:
- [ ] Objetivo é claro em 1 frase.
- [ ] Escopo está limitado e executável.
- [ ] Artefatos estão específicos (caminhos concretos).
- [ ] Riscos têm mitigação prática.
- [ ] Critérios de aceite são verificáveis.
- [ ] Bloco `PROMPT:EXECUTAR` está completo.

Aprovação padrão:

```markdown
✅ ANÁLISE APROVADA

Pode executar conforme PROMPT:EXECUTAR.
Expectativa de PR: [data].
```

Se reprovar:

```markdown
⚠️ ANÁLISE COM AJUSTES

- [ ] ponto 1
- [ ] ponto 2

Revisar o relatório e reenviar para análise.
```

---

## 🚀 Como EXECUTAR (Codex) por tipo de ISSUE

### TIPO = CODE
```markdown
Executar ISSUE-[N] conforme relatorio.

Passos:
1) Implementar somente o escopo definido.
2) Atualizar docstring e documentação de API afetada.
3) Validar:
   - python -m compileall src/
   - python tools/check_api_docs.py --strict
   - python tools/check_links.py docs/ README.md
   - PYTHONPATH=src pytest tests/ -v
4) Atualizar execution_queue e log da issue.
5) Commit + PR ready for review.
```

### TIPO = DOCS
```markdown
Executar ISSUE-[N] conforme relatorio.

Passos:
1) Alterar somente artefatos de documentação listados.
2) Preservar links e navegação.
3) Validar:
   - python tools/check_links.py docs/ README.md
   - python tools/sync_hub.py --check
4) Atualizar execution_queue e log da issue.
5) Commit + PR ready for review.
```

### TIPO = TEST
```markdown
Executar ISSUE-[N] conforme relatorio.

Passos:
1) Criar/ajustar testes previstos no escopo.
2) Rodar testes isolados e suíte geral.
3) Validar:
   - PYTHONPATH=src pytest tests/ -v
4) Atualizar execution_queue e log da issue.
5) Commit + PR ready for review.
```

### TIPO = ADR
```markdown
Executar ISSUE-[N] conforme relatorio.

Passos:
1) Criar/atualizar ADR em docs/governance/adr/.
2) Atualizar docs/governance/adr/INDEX.md.
3) Validar:
   - python tools/check_links.py docs/
   - python tools/sync_hub.py --check
4) Atualizar execution_queue e log da issue.
5) Commit + PR ready for review.
```

---

## ✅ Como FINALIZAR (Humano)

Checklist de fechamento:
- [ ] PR tem evidências de validação.
- [ ] `execution_queue.csv` está atualizado.
- [ ] Log da issue foi atualizado em `prompts/logs/`.
- [ ] Links/documentação não quebraram.
- [ ] HUB consistente (`python tools/sync_hub.py --check`).

Modelo de aprovação:

```markdown
✅ APROVADO

Validações revisadas e rastreabilidade confirmada.
Pode fazer merge.
```

Modelo de ajuste:

```markdown
⚠️ AJUSTES NECESSÁRIOS

1. [ajuste 1]
2. [ajuste 2]

Depois de corrigir, reenviar para revisão.
```

---

## 🔗 Links úteis
- [HUB_CONTROLE.md](../HUB_CONTROLE.md)
- [execution_queue.csv](../execution_queue.csv)
- [Workflow integrado](../WORKFLOW_INTEGRADO.md)
- [relatorios/](./relatorios/)
- [logs/](./logs/)
