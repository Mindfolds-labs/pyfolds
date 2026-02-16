# ISSUE-[N] — [Título Completo]

> **Sistemas/Área:** [docs/código/testes]  
> **Status:** [✅ Concluída | 🔄 Em Progresso | ⏳ Planejada | ❌ Bloqueada]  
> **Sprint:** [1/3 ou N/A]  
> **Data:** [YYYY-MM-DD]  
> **Responsável:** [Nome ou "A definir"]

---

## 📊 Status Executivo

<div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; border-radius: 4px;">

**O Que Já Está Pronto:**
- ✅ [Item 1]
- ✅ [Item 2]

**O Que Ainda Falta (Próximos Passos):**
- ⏳ [Item 1]
- ⏳ [Item 2]

</div>

---

## 🎯 1. Objetivo

[Descrever claramente o objetivo desta issue]

**Exemplo:**
> Padronizar arquivos canônicos na raiz do repositório para melhorar onboarding e conformidade IEEE/ISO.

---

## 📋 2. Escopo

**O que está INCLUÍDO:**
- ✅ Criar `CONTRIBUTING.md` na raiz
- ✅ Criar `CHANGELOG.md` na raiz
- ✅ Preencher `release_process.md`

**O que NÃO está incluído:**
- ❌ Refatorar estrutura de `/docs` (é ISSUE-001)
- ❌ Criar novos ADRs (é demanda diferente)

---

## ✅ 3. O Que Já Está Pronto

### 3.1 Sprint 1 — Gaps Críticos (FECHADO)

<div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 12px; border-radius: 4px;">

**Arquivos Criados:**

✅ **CONTRIBUTING.md** (raiz)
```
- Guia canônico de contribuição
- Ponte para docs/development/CONTRIBUTING.md
- 20 linhas, conciso e direto
```

✅ **CHANGELOG.md** (raiz)
```
- Keep a Changelog format
- Semver versionado
- Seção [Unreleased] + [2.0.0]
```

✅ **docs/development/DEVELOPMENT_HUB.md**
```
- Arquivo de compatibilidade
- Links para HUB_CONTROLE.md
```

✅ **docs/development/release_process.md**
```
- 6 seções: Objetivo, Escopo, Fluxo, Checklist
- Procedimento auditável
```

✅ **src/pyfolds/serialization/foldio.py**
```
- ADR-001/002/003 referenciadas no docstring
- Rastreabilidade melhorada
```

✅ **pyproject.toml**
```
- Novo extra: [project.optional-dependencies] examples
- torchvision>=0.15.0 declarado
```

✅ **docs/ARCHITECTURE.md**
```
- Referência de diagrama atualizada
- docs/diagrams/ → docs/architecture/blueprints/
```

✅ **CI/CD**
```
- .github/workflows/validate-docs.yml criado
- tools/validate_docs_links.py implementado
```

✅ **Sincronização**
```
- execution_queue.csv atualizado
- HUB_CONTROLE.md regenerado
- Links validados
```

</div>

---

### 3.2 Sprint 2 — (EM PLANEJAMENTO)

<div style="background-color: #e2e3e5; border-left: 4px solid #6c757d; padding: 12px; border-radius: 4px;">

⏳ Validação de docs em CI (melhorar)  
⏳ Normalizar `tests/performance/` vs `tests/perf/`  
⏳ Documentar decisão em `docs/development/testing.md`

</div>

---

### 3.3 Sprint 3 — (EM PLANEJAMENTO)

<div style="background-color: #e2e3e5; border-left: 4px solid #6c757d; padding: 12px; border-radius: 4px;">

⏳ Consolidar diagramas em `docs/diagrams/` ou alias  
⏳ Atualizar índices finais (`docs/index.md`, etc)

</div>

---

## ⏳ 4. Próximos Passos

### Para Sprint 2:
- [ ] Expandir validação de docs em GitHub Actions
- [ ] Decidir: `tests/performance/` ou `tests/perf/`?
- [ ] Documentar em `docs/development/testing.md`
- [ ] Atualizar `pyproject.toml` (se necessário)

### Para Sprint 3:
- [ ] Revisar estrutura de diagramas
- [ ] Criar alias se necessário (`docs/diagrams/` → `docs/architecture/blueprints/`)
- [ ] Atualizar `docs/index.md`
- [ ] Publicar v0.1.0

---

## 📝 5. PROMPT PARA EXECUTAR

<!-- PROMPT:INICIO -->

### Contexto
[Seu contexto aqui]

### Instruções Para Codex

Você é um assistente IA ajudando a executar ISSUE-[N].

**Tarefa:** [Descrever o que fazer]

**Arquivos a Alterar:**
1. `arquivo1.md` — [o que fazer]
2. `arquivo2.py` — [o que fazer]
3. `docs/development/execution_queue.csv` — [o que fazer]

**Validações Após Execução:**
```bash
python tools/sync_hub.py --check
python tools/validate_docs_links.py
git status
```

**Commit Final:**
```bash
git add [arquivos]
git commit -m "ISSUE-[N]: [descrição]"
```

<!-- PROMPT:FIM -->

---

## 🔗 6. Referências

| Tipo | Referência |
|------|-----------|
| **ADR** | [ADR-031](../docs/governance/adr/ADR-031-*.md) — Governança operacional |
| **Related ISSUE** | [ISSUE-003](./ISSUE-003-auditoria-completa.md) — Auditoria |
| **Documentation** | [HUB_CONTROLE.md](../docs/development/HUB_CONTROLE.md) |
| **CSV** | [execution_queue.csv](../docs/development/execution_queue.csv) |

---

## 📌 7. Critérios de Aceite

- [x] Objetivo alcançado
- [x] Arquivos criados/atualizados
- [x] Links validados
- [x] Sincronização OK
- [x] Sem erros de sintaxe
- [x] Commit realizado

---

## 📝 Histórico

| Data | Ação | Status |
|------|------|--------|
| 2026-02-16 | Sprint 1 iniciado | ✅ Concluído |
| 2026-02-16 | Commit 2851338 | ✅ Mergeado |
| TBD | Sprint 2 início | ⏳ Planejado |
| TBD | Sprint 3 início | ⏳ Planejado |

---

**Mantido por:** Codex | **Última atualização:** 2026-02-16

