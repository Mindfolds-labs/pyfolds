# PROMPT: Executar ISSUE-[N] (Estágio 3 — EXECUTAR)

## 🚀 Codex Precisa Fazer

### Passo 1: Leia o Relatório
````bash
cat prompts/relatorios/ISSUE-[N]-[slug].md
````

Extraia:
- Objetivo
- Artefatos
- Cronograma
- Riscos

### Passo 2: Execute as Mudanças

Para cada artefato na lista:
````
1. Abra arquivo
2. Faça mudança descrita
3. Salve
4. Continuar
````

### Passo 3: Valide (PMBOK: Monitoramento)
````bash
# Sintaxe
python -m compileall src/

# Links
python tools/validate_docs_links.py

# HUB
python tools/sync_hub.py --check

# Testes
pytest tests/ -v
````

**Esperado:** ✅ Tudo verde

### Passo 4: Atualize CSV
````
docs/development/execution_queue.csv

De:   ISSUE-[N],...,"Planejada",...
Para: ISSUE-[N],...,"Em progresso",...
````

### Passo 5: Faça Commit
````bash
git add [arquivos]
git commit -m "ISSUE-[N]: [descrição]

PMBOK: Execução iniciada
Artefatos: [N] modificados
Validações: ✅ Sintaxe | ✅ Links | ✅ Testes"
````

### Passo 6: Atualize Log
````
prompts/relatorios/ISSUE-[N]-[slug]-LOG.md
````

Adicione seção:
````markdown
## 3️⃣ EXECUTADO (Data/Hora)

**PMBOK Fase:** Execução + Monitoramento

├─ Executor: Codex
├─ Data: [YYYY-MM-DD HH:MM]
├─ Status: 🔄 Em Progresso
├─ Commit: [hash]
├─ Artefatos Alterados: [N]
├─ Validações:
│  ├─ Sintaxe: ✅ OK
│  ├─ Links: ✅ OK
│  ├─ Testes: ✅ [N] passed
│  └─ HUB Sync: ✅ OK
├─ Tempo: [X minutos]
└─ Próximo: Aprovação em PR
````

### Passo 7: Atualize Pull Request
````bash
# No GitHub PR (mude de DRAFT para READY):

Status: **READY FOR REVIEW**

Adicione comentário:
````
## ✅ Execução Completa

**PMBOK: Execução**

- ✅ Commit: [hash]
- ✅ Artefatos: [N] modificados
- ✅ Validações: Todas OK
- ✅ Tempo: [X min]

**Próximo:** Aprovação do usuário (Estágio 4 — FINALIZAR)
````

### Resultado
````
🔄 EXECUTADO

├─ Mudanças: ✅ Feitas
├─ Validações: ✅ OK
├─ Log: ✅ Atualizado
├─ PR: ✅ Ready for Review
└─ Próximo: FINALIZAR
````
````

---

## 4️⃣ ESTÁGIO: FINALIZAR

### PMBOK: Encerramento
- Aceitar entregáveis
- Encerrar projeto
- Lições aprendidas

### 👤 Responsável
**Você (Humano) — Via GitHub PR**

### 📝 Prompt de Finalização
````markdown
# PROMPT: Finalizar ISSUE-[N] (Estágio 4 — FINALIZAR)

## ✅ Você (via GitHub PR) Precisa Fazer

### Passo 1: Leia o PR

No GitHub, vá em:
````
Pull Requests > [ISSUE-[N]] [Tema]
