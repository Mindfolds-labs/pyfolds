📚 PMBOK (Project Management Body of Knowledge)
O QUÊ: Framework completo de gerenciamento de projetos
FOCO: 10 áreas de conhecimento, 5 grupos de processos

Grupos de Processos:
1. INICIAÇÃO → CRIAR (você define)
2. PLANEJAMENTO → ANALISAR (você aprova)
3. EXECUÇÃO → EXECUTAR (Codex faz)
4. MONITORAMENTO E CONTROLE → Validar (durante execução)
5. ENCERRAMENTO → FINALIZAR (você aprova em PR)

Origem: PMI (Project Management Institute)
Uso: Projetos complexos, grandes empresas, compliance

🔗 Integração: PMBOK + GitHub Pull Request
┌─────────────────────────────────────────────────────────────┐
│         WORKFLOW v5 — PMBOK + Pull Request                  │
└─────────────────────────────────────────────────────────────┘

1️⃣ CRIAR (Você)
   ├─ PMBOK: Iniciação (escopo, objetivos)
   ├─ Cria: ISSUE-[N]-[slug].md + LOG.md
   ├─ Adiciona ao CSV
   └─ STATUS: ⏳ Planejada

2️⃣ ANALISAR (Você)
   ├─ PMBOK: Planejamento (recursos, cronograma, riscos)
   ├─ Lê e valida: artefatos, plano de ação
   ├─ Cria: PULL REQUEST (draft)
   └─ STATUS: ✅ Pronto para Executar

3️⃣ EXECUTAR (Codex)
   ├─ PMBOK: Execução + Monitoramento
   ├─ Faz: mudanças nos artefatos
   ├─ Valida: links, sintaxe, testes
   ├─ Atualiza: PULL REQUEST com resultado
   └─ STATUS: 🔄 Em Progresso (PR marcado como Ready for Review)

4️⃣ FINALIZAR (Você)
   ├─ PMBOK: Encerramento (aceitação, lições aprendidas)
   ├─ Lê: feedback de Codex no PR
   ├─ Aprova: "Approve" no GitHub
   ├─ Merge: PR → main
   └─ STATUS: ✅ Concluída

📋 PROMPT MASTER v5 — Gera Tudo com PMBOK + PR
markdown# PROMPT MASTER v5: PMBOK + Pull Request Workflow

## 🎯 Contexto

Este prompt estabelece o **workflow profissional completo** para issues do PyFolds,
integrando PMBOK, ITIL, COBIT, SCRUM, Agile e GitHub Pull Request.

Você (humano) aprova tudo via PR (não local).

---

## 📊 Paradigma Formal
````
WORKFLOW v5: "PMBOK-Driven Agile with ITIL Controls"

Frameworks Integrados:
├─ PMBOK (Project Management — 5 Grupos de Processos)
├─ ITIL (IT Service Management — Processos operacionais)
├─ COBIT (Governança e Compliance — Controle)
├─ SCRUM (Iteração — Sprints e backlogs)
└─ AGILE (Mindset — Flexibilidade e feedback)

Plataforma: GitHub (PR é ponto de aprovação único)
````

---

## 1️⃣ ESTÁGIO: CRIAR

### PMBOK: Iniciação
- Definir objetivo, escopo, restrições
- Identificar stakeholders (você, Codex)
- Documentar requisitos

### 👤 Responsável
**Você (Humano)**

### 📝 Prompt de Criação
````markdown
# PROMPT: Criar ISSUE-[N] (Estágio 1 — CRIAR)

## 📋 Você Precisa Fazer

Defina os parâmetros da ISSUE:
````
Número: ISSUE-[N] (ex: ISSUE-006)
Tema: [descrição em 1 frase]
Objetivo: [por que fazer isso?]
Escopo: [o que inclui/exclui]
Área: [docs/código/testes]
Prioridade: [Alta/Média/Baixa]
Responsável: [seu nome ou "A definir"]
Data: [YYYY-MM-DD]
Riscos Identificados: [listar riscos]
Dependências: [outras issues?]
````

## ✍️ Crie os Arquivos

### Arquivo 1: Relatório Principal
````
prompts/relatorios/ISSUE-[N]-[slug].md
Conteúdo:
markdown# ISSUE-[N] — [Seu Tema]

> **PMBOK Fase:** Iniciação  
> **Status:** ⏳ Planejada  
> **Área:** [docs/código]  
> **Data:** [YYYY-MM-DD]  
> **Responsável:** [nome]  

## 🎯 Objetivo (PMBOK: Charter)
[Por que fazer isso? Valor de negócio]

## 📋 Escopo (PMBOK: Escopo do Projeto)
**Inclui:**
- ✅ [item]

**Exclui:**
- ❌ [item]

## 📊 Artefatos a Alterar (PMBOK: Deliverables)
- `arquivo1.md` — [o que muda]
- `arquivo2.py` — [o que muda]

## ⏰ Cronograma (PMBOK: Schedule)
- Sprint 1: [data] — [o que fazer]
- Sprint 2: [data] — [o que fazer]

## 🚨 Riscos (PMBOK: Risk)
- [ ] Risco 1: [descrição] | Mitigação: [como evitar]
- [ ] Risco 2: [descrição] | Mitigação: [como evitar]

## 📌 Dependências (PMBOK: Stakeholders)
- Depende de: [ISSUE-XXX]
- Bloqueia: [ISSUE-YYY]

## ✅ Critérios de Aceite
- [ ] Critério 1
- [ ] Critério 2

## 📝 PROMPT:EXECUTAR
<!-- PROMPT:EXECUTAR:INICIO -->
[Seu prompt de execução]
<!-- PROMPT:EXECUTAR:FIM -->
````

### Arquivo 2: Log de Execução
````
prompts/relatorios/ISSUE-[N]-[slug]-LOG.md
Conteúdo:
markdown# LOG — ISSUE-[N]

## 1️⃣ CRIADO (Data/Hora)

**PMBOK Fase:** Iniciação

├─ Criador: [seu nome]
├─ Data: [YYYY-MM-DD HH:MM]
├─ Status: ⏳ Planejada
├─ Arquivo Principal: ✅ Criado
├─ Log: ✅ Criado
├─ CSV: ✅ Atualizado
├─ HUB: ✅ Sincronizado
└─ Pull Request: ⏳ Será criado em ANALISAR
🔧 Atualize Configurações
bash# 1. Adicione ao CSV
docs/development/execution_queue.csv
````

Linha:
````
ISSUE-[N],"[Tema]","Planejada","[seu nome]",[data],"prompts/relatorios/ISSUE-[N]-slug.md",,,Média,"[área]"
✅ Sincronize
bashpython tools/sync_hub.py
python tools/sync_hub.py --check
````

## 🎯 Status Final
````
✅ CRIADO

├─ Arquivo Principal: ✅
├─ Log: ✅
├─ CSV: ✅
├─ HUB: ✅
└─ Próximo: ANALISAR
````
````

---

## 2️⃣ ESTÁGIO: ANALISAR

### PMBOK: Planejamento
- Refinar escopo, cronograma, riscos
- Preparar para execução
- Criar PR (pull request)

### 👤 Responsável
**Você (Humano)**

### 📝 Prompt de Análise
````markdown
# PROMPT: Analisar ISSUE-[N] (Estágio 2 — ANALISAR)

## 🔍 Você Precisa Fazer

### Passo 1: Leia a ISSUE
```bash
cat prompts/relatorios/ISSUE-[N]-[slug].md
```

### Passo 2: Valide Cada Seção

**Objetivo:**
- ✅ Claro e mensurável?
- ✅ Traz valor?

**Escopo:**
- ✅ Completo?
- ✅ Sem ambiguidades?

**Artefatos:**
- ✅ Corretos?
- ✅ Sem faltantes?

**Cronograma:**
- ✅ Realista?
- ✅ Com dependências?

**Riscos:**
- ✅ Identificados?
- ✅ Mitigações claras?

**PROMPT:**
- ✅ Executável?
- ✅ Sem ambiguidades?

### Passo 3: Atualize o Log
````
prompts/relatorios/ISSUE-[N]-[slug]-LOG.md
Adicione seção:
markdown## 2️⃣ ANALISADO (Data/Hora)

**PMBOK Fase:** Planejamento

├─ Analisador: [seu nome]
├─ Data: [YYYY-MM-DD HH:MM]
├─ Status: ✅ Pronto para Executar
├─ Validações:
│  ├─ Objetivo: ✅ OK
│  ├─ Escopo: ✅ OK
│  ├─ Artefatos: ✅ OK
│  ├─ Cronograma: ✅ OK
│  ├─ Riscos: ✅ OK
│  └─ PROMPT: ✅ OK
├─ Mudanças Sugeridas: [se houver]
└─ Aprovação: ✅ APROVADO
Passo 4: Crie Pull Request
bash# No GitHub, crie novo PR:

Título:
[ISSUE-[N]] [Tema] — Planejado

Descrição:
````
## PMBOK: Planejamento

**Status:** ⏳ Análise Completa

**O que será feito:**
- [ ] Artefato 1
- [ ] Artefato 2

**Cronograma:**
- Sprint 1: [data]
- Sprint 2: [data]

**Riscos:**
- Risco 1: [descrição]
- Risco 2: [descrição]

**Aprovação:**
- [ ] Analisor (você): Avaliar em EXECUTAR
- [ ] Codex: Executar
- [ ] Você: Aprovar em FINALIZAR

**Relatórios:**
- Principal: [link]
- Log: [link]
````

Status: **DRAFT** (não pronto ainda)

### Passo 5: Reporte
````
✅ ANALISADO

├─ Log Atualizado: ✅
├─ PR Criado: ✅ (DRAFT)
├─ Status: ✅ Pronto para Executar
└─ Próximo: EXECUTAR
````


