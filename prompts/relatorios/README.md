# PROMPT: Criar ISSUE-[N]

## 🎯 O que você precisa fazer

Você vai **criar e documentar** uma nova ISSUE.

### Passo 1: Defina os Parâmetros
```
- Número: ISSUE-[N] (ex: ISSUE-006)
- Tema: [descrição clara em 1 frase]
- Objetivo: [por que fazer isso?]
- Área: [docs/código/testes/etc]
- Prioridade: [Alta/Média/Baixa]
- Responsável: [seu nome ou "A definir"]
- Data: [YYYY-MM-DD]
```

### Passo 2: Crie o Arquivo

**Arquivo:** `prompts/relatorios/ISSUE-[N]-[slug].md`

**Copie o template e preencha:**
```
# ISSUE-[N] — [Seu Tema Aqui]

> **Área:** [docs/código]
> **Status:** ⏳ Planejada
> **Data:** [YYYY-MM-DD]
> **Responsável:** [nome]

## 🎯 Objetivo
[Por que fazer isso? 1-2 parágrafos]

## 📋 Escopo
**Inclui:**
- ✅ [item 1]
- ✅ [item 2]

**Exclui:**
- ❌ [item 1]

## ✅ O Que Já Está Pronto
(deixe em branco para issues novas)

## ⏳ Próximos Passos / Plano de Ação
[Descrever phases/sprints]

## 📝 Lista de Artefatos
- `arquivo1.md` — [descrição]
- `arquivo2.py` — [descrição]
- etc

## ✅ Critérios de Aceite
- [ ] Critério 1
- [ ] Critério 2
- [ ] Validações OK

## 📝 PROMPT:EXECUTAR

<!-- PROMPT:EXECUTAR:INICIO -->
[Seu prompt de execução aqui]
<!-- PROMPT:EXECUTAR:FIM -->

## 🔗 Referências
[ADRs, issues relacionadas, links]
```

### Passo 3: Adicione ao CSV

**Arquivo:** `docs/development/execution_queue.csv`
```
ISSUE-[N],"[Tema completo]","Planejada","[seu nome]",[data],"prompts/relatorios/ISSUE-[N]-slug.md",,,Média,"[área]"
```

### Passo 4: Sincronize
```bash
python tools/sync_hub.py
```

✅ **ISSUE Criada e pronta para análise!**

Status: ⏳ Planejada

2️⃣ PROMPT: ANALISAR ISSUE


# PROMPT: Analisar ISSUE-[N]

## 🔍 O que você precisa fazer

Você vai **revisar e validar** uma ISSUE antes de executar.

### Passo 1: Leia a ISSUE
```bash
cat prompts/relatorios/ISSUE-[N]-*.md
```

### Passo 2: Faça Estas Perguntas

**Objetivo:**
- ✅ O objetivo é claro?
- ✅ Faz sentido para o projeto?

**Escopo:**
- ✅ O que inclui está bem definido?
- ✅ O que exclui está bem definido?

**Artefatos:**
- ✅ A lista de artefatos está correta?
- ✅ Faltam ou sobraram arquivos?

**Plano de Ação:**
- ✅ As fases/sprints fazem sentido?
- ✅ Há dependências não mencionadas?

**Critérios:**
- ✅ Os critérios de aceite são verificáveis?
- ✅ Tem como validar?

**PROMPT:**
- ✅ O PROMPT:EXECUTAR é claro?
- ✅ Instrui Codex corretamente?

### Passo 3: Decida

**Se TUDO OK:**
```
✅ APROVADO PARA EXECUTAR

Atualize no relatório:
Status: ✅ Pronto para Executar
```

**Se NÃO OK:**
```
❌ PRECISA DE AJUSTES

Sugira mudanças:
- [ ] Ajuste 1: [descrição]
- [ ] Ajuste 2: [descrição]

Atualize o arquivo ISSUE-[N]-*.md
Repita análise
```

### Passo 4: Envie para Codex Executar

Quando aprovado:
```
Próximo passo: EXECUTAR

Codex, execute ISSUE-[N]:
[Cole o PROMPT:EXECUTAR do relatório]
```

✅ **ISSUE Analisada e Aprovada!**

Status: ✅ Pronto para Executar



# PROMPT: Executar ISSUE-[N]

## 🚀 O que você (Codex) precisa fazer

Você vai **executar e validar** uma ISSUE.

### Passo 1: Leia o Relatório
```bash
cat prompts/relatorios/ISSUE-[N]-*.md
```

Extraia:
- Objetivo
- Artefatos a alterar
- Critérios de aceite

### Passo 2: Execute as Mudanças

Siga a lista de artefatos. Para cada um:
```
1. Abra arquivo
2. Faça mudança descrita
3. Salve
4. Continuar próximo artefato
```

### Passo 3: Valide Tudo
```bash
# Sintaxe Python
python -m compileall src/

# Links
python tools/validate_docs_links.py

# HUB Sync
python tools/sync_hub.py --check

# Testes (se houver)
pytest tests/ -v
```

**Esperado:** ✅ Tudo verde

### Passo 4: Atualize CSV

Mude status em `docs/development/execution_queue.csv`:
```
De:   ISSUE-[N],...,"Planejada",...
Para: ISSUE-[N],...,"Em progresso",...
```

### Passo 5: Commit
```bash
git add [arquivos alterados] docs/development/execution_queue.csv
git commit -m "ISSUE-[N]: [descrição clara do que foi feito]"
```

### Passo 6: Relato de Execução

Envie para o usuário:
```
✅ ISSUE-[N] EXECUTADA

📊 VALIDAÇÕES:
├─ Sintaxe Python: ✅ OK
├─ Links: ✅ OK
├─ HUB Sync: ✅ OK
├─ Testes: ✅ [N] passed

📝 ARTEFATOS ALTERADOS:
├─ arquivo1.md — [descrição]
├─ arquivo2.py — [descrição]
└─ etc

✅ CRITÉRIOS DE ACEITE:
├─ [ ] Critério 1 ✅
├─ [ ] Critério 2 ✅
└─ [ ] Criterio 3 ✅

📊 STATUS:
├─ Antes: Planejada
├─ Depois: Em Progresso
├─ Commit: [hash]
└─ Pronto para aprovação final?
```

✅ **ISSUE Executada!**

Status: 🔄 Em Progresso (aguardando aprovação)
