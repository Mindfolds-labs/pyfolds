# 📊 Relatórios de Auditoria — PyFolds

> Esta pasta contém relatórios gerados por prompts de auditoria antes da execução de melhorias.

## 🎯 Objetivo

Manter um histórico de diagnósticos técnicos para:

- ✅ Rastrear decisões
- ✅ Documentar problemas encontrados
- ✅ Basear futuras issues e PRs
- ✅ Facilitar revisão humana antes da execução

---

## 📂 Estrutura dos Arquivos

- `template_relatorio.md` → Modelo padrão para novos relatórios
- `ISSUE-XXX-descricao.md` → Relatórios nomeados por issue relacionada

---

## 🚀 Como Usar

### 1️⃣ PROMPT: SEGUIR (Acompanhar Issue Existente)

<!-- PROMPT:SEGUIR:INICIO -->

#### Contexto
Você é um assistente IA ajudando a acompanhar o progresso de uma issue já criada.

#### Tarefa: Visualizar Status de Uma Issue

**Comando:**
```bash
# Passo 1: Ver qual issue acompanhar
ls prompts/relatorios/ISSUE-*.md

# Passo 2: Ler o relatório
cat prompts/relatorios/ISSUE-005-plano-acao-consolidacao.md

# Passo 3: Verificar status no CSV
grep "ISSUE-005" docs/development/execution_queue.csv
```

**Para Codex/IA:**
```markdown
Usuário quer acompanhar ISSUE-[N]. Faça:

1. **Leia** `prompts/relatorios/ISSUE-[N]-*.md`
2. **Extraia** (em formato tabela):
   - Objetivo
   - Status atual
   - O que já foi feito (✅)
   - O que falta (⏳)
   - Próximos passos

3. **Valide** em `docs/development/execution_queue.csv`:
   - Status oficial
   - Responsável
   - Data

4. **Mostre** ao usuário (resumo executivo)
```

**Exemplo de Saída:**
```
✅ ISSUE-005 — Consolidação Total

Status Oficial: Em Progresso (Parcial)
Sprint: 1/3
Responsável: Codex
Data: 2026-02-16

✅ O Que Já Está Pronto:
├─ CONTRIBUTING.md
├─ CHANGELOG.md
├─ docs/development/release_process.md
└─ ... (8 artefatos)

⏳ Próximos Passos:
├─ Sprint 2: Validação de docs em CI
├─ Sprint 2: Normalizar testes
└─ Sprint 3: Consolidar diagramas
```

<!-- PROMPT:SEGUIR:FIM -->

---

### 2️⃣ PROMPT: CRIAR (Criar Nova Issue + Relatório)

<!-- PROMPT:CRIAR:INICIO -->

#### Contexto
Você é um assistente IA ajudando a criar uma **nova ISSUE** no PyFolds.

#### Tarefa: Planejar Nova Issue

**Informações que Você (Humano) Deve Fornecer:**
```
- Número da issue: ISSUE-[N]
- Tema/Título: [descrição clara]
- Objetivo: [por que fazer isso?]
- Área: [docs/código/testes/etc]
- Prioridade: [Alta/Média/Baixa]
- Responsável: [nome ou "A definir"]
- Data: [YYYY-MM-DD]
```

**Para Codex/IA:**

Você recebe as informações acima e faz:
```bash
# 1. Criar linha no CSV
docs/development/execution_queue.csv
├─ Adicione: ISSUE-[N],"Tema completo","Planejada","A definir",[data],"prompts/relatorios/ISSUE-[N]-slug.md"

# 2. Criar relatório
prompts/relatorios/ISSUE-[N]-slug.md
├─ Copie template_relatorio.md
├─ Preencha:
│  ├─ Cabeçalho (ID, Status, Área, Data)
│  ├─ Objetivo (1-2 parágrafos)
│  ├─ Escopo (o que inclui/exclui)
│  ├─ Artefatos a alterar (lista)
│  ├─ Próximos passos (roadmap)
│  └─ PROMPT:EXECUTAR (copiar de baixo)

# 3. Sincronizar
python tools/sync_hub.py
python tools/sync_hub.py --check

# 4. Validar
python tools/validate_docs_links.py

# 5. Commit
git add docs/development/execution_queue.csv prompts/relatorios/ISSUE-[N]-slug.md
git commit -m "ISSUE-[N]: criar planejamento"
```

**Exemplo:**
```markdown
# ISSUE-006 — Normalizar Estrutura de Testes

> **Área:** Desenvolvimento/Testes  
> **Status:** ⏳ Planejada  
> **Data:** 2026-02-16  
> **Responsável:** A definir

## 🎯 Objetivo
Decidir e normalizar: tests/performance/ ou tests/perf/?

## 📋 Escopo
- Decidir padrão
- Documentar em docs/development/testing.md
- Refatorar diretório
- Testes verdes

[... resto do relatório ...]
```

<!-- PROMPT:CRIAR:FIM -->

---

### 3️⃣ PROMPT: EXECUTAR (Rodar Issue Planejada)

<!-- PROMPT:EXECUTAR:INICIO -->

#### Contexto
Você é um assistente IA ajudando a **executar uma ISSUE planejada** no PyFolds.

#### Tarefa: Executar Issue Passo-a-Passo

**Informações que Você (Humano) Deve Fornecer:**
```
- Número da issue: ISSUE-[N]
- Qual relatório: prompts/relatorios/ISSUE-[N]-*.md
- Contexto adicional: [se houver]
```

**Para Codex/IA:**
```bash
# 1. Ler o relatório
cat prompts/relatorios/ISSUE-[N]-*.md

# 2. Encontre a seção "PROMPT:EXECUTAR"
# (está no próprio relatório, entre comentários HTML)

# 3. Extraia e siga as instruções ali

# 4. Estrutura típica:
├─ Ler objetivo + escopo
├─ Identificar artefatos a alterar
├─ Executar mudanças
├─ Validar (testes, links, sintaxe)
├─ Atualizar CSV (status: "Em progresso" → "Concluída")
├─ Sincronizar HUB
└─ Commit final

# 5. Exemplo de execução:
python tools/sync_hub.py --check
python tools/validate_docs_links.py
pytest tests/ -v
git status
git add [arquivos alterados]
git commit -m "ISSUE-[N]: [descrição do que foi feito]"
```

**Fluxo Esperado:**
```
ISSUE-[N] (Planejada)
    ↓ (Humano copia PROMPT:EXECUTAR)
Codex executa
    ↓ (Humano revisa)
Feedback humano
    ↓ (Se OK)
Commit + Merge
    ↓ (Automation)
CSV atualizado → "Concluída"
HUB sincronizado automaticamente ✅
```

<!-- PROMPT:EXECUTAR:FIM -->

---

## 📊 Fluxo Completo (Visual)
```
┌─────────────────────────────────────────────────────────────┐
│                     CICLO DE UMA ISSUE                       │
└─────────────────────────────────────────────────────────────┘

1️⃣ CRIAR (Humano + Codex)
   ├─ Humano fornece: número, tema, objetivo, área
   ├─ Codex cria: CSV + relatório + sincroniza
   └─ Resultado: ISSUE-[N] em "Planejada"

2️⃣ REVISAR (Humano)
   ├─ Humano lê: prompts/relatorios/ISSUE-[N]-*.md
   ├─ Humano aprova: objetivo, escopo, artefatos
   └─ Resultado: Issue aprovada (pronta para executar)

3️⃣ EXECUTAR (Codex)
   ├─ Humano copia: PROMPT:EXECUTAR do relatório
   ├─ Codex executa: mudanças, testes, validações
   ├─ Codex relata: o que foi feito, evidências
   └─ Resultado: Artefatos alterados + validados

4️⃣ AVALIAR (Humano)
   ├─ Humano verifica: testes verdes, links OK, sintaxe OK
   ├─ Humano aprova ou pede ajustes
   └─ Resultado: ✅ Pronto para merge ou ❌ Voltar para step 3

5️⃣ FINALIZAR (Automação)
   ├─ Humano aprova merge
   ├─ Automation: sincroniza HUB (CSV → atualiza status)
   └─ Resultado: ISSUE-[N] em "Concluída"
```

---

## 🎯 Padrão de Nomes de Relatórios
```
ISSUE-[N]-[slug].md

Exemplos:
├─ ISSUE-001-reestruturacao-docs.md
├─ ISSUE-005-plano-acao-consolidacao.md
├─ ISSUE-006-normalizar-testes.md
└─ ISSUE-007-refactor-hub-visual.md
```

---

## 🔗 Links Importantes

- **Fila Principal:** [`docs/development/HUB_CONTROLE.md`](../docs/development/HUB_CONTROLE.md)
- **CSV de Execução:** [`docs/development/execution_queue.csv`](../docs/development/execution_queue.csv)
- **Template:** [`template_relatorio.md`](./template_relatorio.md)

---

## ✅ Checklist Pós-Criação/Execução

- [ ] Relatório criado/atualizado
- [ ] CSV sincronizado
- [ ] HUB regenerado
- [ ] Links validados
- [ ] Sem erros de sintaxe
- [ ] Commit realizado

---

**Última atualização:** 2026-02-16  
**Mantido por:** Codex (PyFolds Team)
```

---

## 🎯 **Feedback Externo (Meu Parecer como IA)**

### ✅ O Que Está Ótimo
```
🟢 ESTRUTURA CLARA
   └─ 3 prompts separados (Seguir, Criar, Executar)
   └─ Cada um com contexto + tarefa + exemplo

🟢 FLUXO INTUITIVO
   └─ Humano → Codex → Humano → Merge
   └─ Feedback loop bem definido

🟢 ESCALÁVEL
   └─ Funciona para ISSUE-006, 007, 008... sem mudanças
   └─ Template reutilizável

🟢 RASTREABILIDADE
   └─ Cada ISSUE tem relatório próprio
   └─ CSV é fonte de verdade
   └─ HUB sincroniza automaticamente
```

---

### ⚠️ Sugestões de Melhoria
```
🟡 ADICIONAR: Versionamento de Relatórios
   └─ Quando executar, criar: ISSUE-005-v1.0.md, v1.1.md, etc

🟡 ADICIONAR: Templates de Feedback
   └─ Quando humano avalia, deixar espaço para: ✅/❌/🔴

🟡 ADICIONAR: Checklist de Validação
   └─ Pré-execução: verificar dependências
   └─ Pós-execução: verificar critérios de aceite

🟡 CONSIDERAR: Integração com GitHub Issues
   └─ Adicionar link para PR/Issue oficial do GitHub
```

---

### 💡 **Minha Recomendação**
```
PRÓXIMO PASSO IDEAL:

1️⃣ Usar este novo README.md (com 3 prompts) ✅
2️⃣ Criar ISSUE-006 usando PROMPT:CRIAR
3️⃣ Executar ISSUE-005 Sprint 1 usando PROMPT:EXECUTAR
4️⃣ Você avalia e aprova
5️⃣ Codex faz ajustes se necessário
6️⃣ Merge + Automation atualiza CSV

GANHOS:
├─ Workflow claro e repetível
├─ Rastreabilidade 100%
├─ Fácil de ensinar a novos contribuidores
└─ Escalável para múltiplas issues simultâneas

