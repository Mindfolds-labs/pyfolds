<!-- 
================================================================================
ARQUIVO: README-WORKFLOW-BLOCO.md
Workflow v5 — PMBOK + Pull Request

INSTRUÇÕES DE USO:
1. Copie TODO o conteúdo desta seção (a partir do ## 🔄 Workflow)
2. Abra o README.md do repositório PyFolds
3. Encontre um espaço vazio ou antes da última seção
4. Cole TODO o conteúdo aqui (Ctrl+A, Ctrl+C, Ctrl+V)
5. Salve o arquivo (Ctrl+S)
6. Pronto! Os blocos vão aparecer no navegador

NÃO APAGUE NADA, APENAS COLE
================================================================================
-->

## 🔄 Workflow v5 — PMBOK + Pull Request

### Como Criar e Executar uma ISSUE

Seguimos **PMBOK + GitHub PR** (você aprova tudo via PR):

```
1️⃣ CRIAR (Você)          2️⃣ ANALISAR (Você)      3️⃣ EXECUTAR (Codex)    4️⃣ FINALIZAR (Você)
   ⏳ Planejada  →         ✅ Pronto        →       🔄 Progresso    →       ✅ Concluída
   
   Arquivo:              PR: Draft            Commit              PR: Approve
   ISSUE-[N].md          Ready              Validações          Merge
   LOG.md                                    Log Update
```

### Template Copiável (CRIAR + ANALISAR)

Cole este template ao criar uma ISSUE:

```
# ISSUE-[N] — [Seu Título]

> **PMBOK Fase:** Iniciação  
> **Status:** ⏳ Planejada  
> **Área:** [docs/código/testes]  
> **Data:** [YYYY-MM-DD]  
> **Responsável:** [seu nome]

## 🎯 Objetivo
[Por quê fazer isso?]

## 📋 Escopo
**Inclui:**
- ✅ [item 1]
- ✅ [item 2]

**Exclui:**
- ❌ [item 1]

## 📊 Artefatos
- `arquivo1.md` — [o que muda]
- `arquivo2.py` — [o que muda]

## ⏰ Cronograma (Sprints)
- Sprint 1: [data] — [o que fazer]
- Sprint 2: [data] — [o que fazer]

## 🚨 Riscos & Mitigação
- [ ] Risco 1 | Mitiga: [como evitar]
- [ ] Risco 2 | Mitiga: [como evitar]

## ✅ Critérios de Aceite
- [ ] Critério 1
- [ ] Critério 2

## 📝 PROMPT:EXECUTAR
<!-- PROMPT:EXECUTAR:INICIO -->
[Seu prompt aqui]
<!-- PROMPT:EXECUTAR:FIM -->
```

### Passos Rápidos

**1️⃣ Criar:**

```bash
# 1. Crie arquivo com template acima
nano prompts/relatorios/ISSUE-[N]-[slug].md

# 2. Crie LOG
nano prompts/relatorios/ISSUE-[N]-[slug]-LOG.md

# 3. Adicione ao CSV
nano docs/development/execution_queue.csv
# ISSUE-[N],"Tema","Planejada","[seu nome]",[data],"prompts/relatorios/ISSUE-[N]-slug.md"

# 4. Sincronize
python tools/sync_hub.py
```

**2️⃣ Analisar:**

```bash
# Leia o arquivo criado
cat prompts/relatorios/ISSUE-[N]-[slug].md

# Valide: objetivo ✅, escopo ✅, artefatos ✅, PROMPT ✅

# Crie PR no GitHub (Draft)
# Título: [ISSUE-[N]] [Tema] — Planejado
# Status: DRAFT (não pronto)
```

**3️⃣ Executar (Codex faz):**

```bash
# Codex:
# 1. Lê o relatório
# 2. Faz mudanças nos artefatos
# 3. Valida (sintaxe, links, testes)
# 4. Atualiza PR → "Ready for Review"
```

**4️⃣ Finalizar (Você):**

```bash
# Você:
# 1. Revisa PR no GitHub
# 2. Clica "Approve"
# 3. Clica "Merge"
# (CSV se atualiza automaticamente)
```

### Frameworks Integrados

| Fase | PMBOK | ITIL | COBIT | SCRUM |
|------|-------|------|-------|-------|
| CRIAR | Iniciação | RFC | Objetivo | Planning |
| ANALISAR | Planejamento | CAB | Conformidade | Refinement |
| EXECUTAR | Execução | Implementação | Controle | Sprint |
| FINALIZAR | Encerramento | Auditoria | Compliance | Review |

### Links Importantes

- **Fila:** [`docs/development/HUB_CONTROLE.md`](docs/development/HUB_CONTROLE.md)
- **Relatórios:** [`prompts/relatorios/`](prompts/relatorios/)
- **CSV:** [`docs/development/execution_queue.csv`](docs/development/execution_queue.csv)

---

<!-- 
FIM DO BLOCO

Você colou tudo acima no README.md?
✅ SIM? Salve o arquivo e pronto!
❌ NÃO? Copie novamente, linha por linha, sem apagar nada.

Dúvidas? Veja prompts/relatorios/README.md
-->
