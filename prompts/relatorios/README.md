# PROMPT: Executar ISSUE-[N] — [Seu Título]

## 📋 Contexto

Estou executando **ISSUE-[N]** no repositório PyFolds.

**Objetivo:** [Descrever o objetivo em 1 frase]

**Exemplo:** "Normalizar estrutura de testes: tests/performance/ → tests/perf/"

---

## 🎯 Instruções para Execução

### Passo 1: Ler o Relatório
```bash
cat prompts/relatorios/ISSUE-[N]-[slug].md
```

Extraia:
- Objetivo
- Escopo (o que inclui/exclui)
- Artefatos a alterar
- Critérios de aceite

### Passo 2: Listar Artefatos a Alterar

**Arquivos que você vai mexer:**
- `arquivo1.md` — [o que fazer]
- `arquivo2.py` — [o que fazer]
- `arquivo3.toml` — [o que fazer]
- etc

### Passo 3: Executar Mudanças

Faça as alterações necessárias nos arquivos listados acima.

**Validações intermediárias:**
```bash
# Se mexeu em Python:
python -m compileall src/

# Se mexeu em Markdown:
python tools/validate_docs_links.py

# Se mexeu em dependências:
grep -r "import X" src/ | head -5
```

### Passo 4: Atualizar CSV

**Arquivo:** `docs/development/execution_queue.csv`

Encontre a linha com `ISSUE-[N]` e mude:
```
De:  ISSUE-[N],...,"Planejada",...
Para: ISSUE-[N],...,"Em progresso",...
```

(Se executar completamente, mude para `"Concluída"`)

### Passo 5: Sincronizar HUB
```bash
python tools/sync_hub.py
python tools/sync_hub.py --check
```

**Esperado:** Verde (sem erros)

### Passo 6: Validar Tudo
```bash
# Links
python tools/validate_docs_links.py

# Sintaxe Python (se houver)
python -m compileall src/pyfolds/

# Git status
git status

# Diff (verificar mudanças)
git diff --stat
```

**Esperado:** Sem erros, apenas arquivos alterados

### Passo 7: Commit
```bash
# Adicionar arquivos
git add docs/development/execution_queue.csv prompts/relatorios/ISSUE-[N]-[slug].md [outros arquivos alterados]

# Commit com mensagem clara
git commit -m "ISSUE-[N]: [descrição clara do que foi feito]

Artefatos alterados:
- arquivo1.md
- arquivo2.py
- etc

Validações:
- ✅ Links validados
- ✅ Sincronização OK
- ✅ Sem erros de sintaxe
- ✅ Critérios de aceite atendidos"
```

### Passo 8: Reportar Resultado

**Mostre ao usuário (humano):**
```
✅ ISSUE-[N] Executada com Sucesso!

Status: Em Progresso (ou Concluída)
Commit: [hash do commit]
Artefatos Alterados: [número]
Validações: ✅ Todas Verdes

O Que Foi Feito:
├─ [artefato 1] — [o que mudou]
├─ [artefato 2] — [o que mudou]
└─ [artefato 3] — [o que mudou]

Próximos Passos:
└─ [Humano avalia e aprova para merge]
```

---

## 📌 Notas Importantes

- ✅ Se houver erro em qualquer passo, PARE e reporte ao usuário
- ✅ Não force commit se validações falharem
- ✅ Se precisar fazer ajustes, comunique ao usuário antes de committar
- ✅ Mantenha CSV sempre sincronizado com HUB
- ✅ Deixe um commit por mudança importante (não misture)

---

## 🎯 Critério de Sucesso

- [ ] Relatório lido e compreendido
- [ ] Artefatos identificados
- [ ] Mudanças executadas
- [ ] CSV atualizado
- [ ] HUB sincronizado (python tools/sync_hub.py --check)
- [ ] Links validados (python tools/validate_docs_links.py)
- [ ] Sem erros de sintaxe
- [ ] Commit realizado com mensagem clara
- [ ] Resultado reportado ao usuário

---

**Fim do Prompt.**
