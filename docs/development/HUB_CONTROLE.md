# 🚀 HUB_CONTROLE — Centro de Governança PyFolds

<p align="center">
  <img src="pyfoldnovo.PNG" width="550" alt="PyFolds Logo Banner">
</p>

<p align="center">
  <a href="#id-do-documento">
    <img src="https://img.shields.io/badge/DOC_ID-DEV--HUB--CTRL--001-0A3069?style=for-the-badge&logo=target" alt="ID">
  </a>
  <a href="../../actions">
    <img src="https://img.shields.io/badge/CI_STATUS-Ativo-FFD700?style=for-the-badge&logo=github-actions&logoColor=0A3069" alt="Status">
  </a>
  <a href="#7-referências">
    <img src="https://img.shields.io/badge/STANDARDS-IEEE_|_ISO-0A3069?style=for-the-badge&logo=bookstack" alt="Normas">
  </a>
</p>

> [!IMPORTANT]
> **GOVERNANÇA AUTOMATIZADA:** Esta fila de execução é sincronizada dinamicamente a partir de `docs/development/execution_queue.csv`. Toda alteração deve seguir a norma **IEEE 730**.

---

## 1. Objetivo
Centralizar a fila de execução de documentação e governança para evitar conflitos entre agentes e manter rastreabilidade.

## 2. Escopo e Navegação
Este HUB **não é documentação de usuário final**. Ele deve ser usado apenas por quem mantém a base documental e os artefatos de governança.

| Componente | Link |
| :--- | :--- |
| 🛠️ Dev Index | [`DEVELOPMENT.md`](DEVELOPMENT.md) |
| 📜 Master Plan | [`../governance/MASTER_PLAN.md`](../governance/MASTER_PLAN.md) |
| ⚖️ ADR Index | [`../governance/adr/INDEX.md`](../governance/adr/INDEX.md) |

---

## 3. Regras Operacionais
1. Toda issue deve referenciar uma ADR quando alterar arquitetura ou processo.
2. Apenas uma issue em estado **Em Progresso** por agente.
3. Mudanças em `/docs/governance` exigem atualização de índices.

---

## 4. Fila de Execução (Tabela Resumida)

| ID | Status | Tema | Responsável | Data |
| :-- | :-- | :-- | :-- | :-- |
| ISSUE-023 | ✅ Concluída | Auditoria corretiva de estabilidade runtime | Codex | 2026-02-17 |
| ISSUE-021 | ⏳ Planejada | Auditoria total do repositório (Análise Sênior) | Codex | 2026-02-17 |
| ISSUE-020 | ✅ Concluída | Relatório CI Docs Hub (Sphinx/MyST) | Codex | 2026-02-17 |
| ISSUE-012 | ✅ Concluída | Auditoria de código em src + testes + ADR-035 | Codex | 2026-02-17 |
| ISSUE-001 | ✅ Concluída | Reestruturação sistêmica de /docs e raiz | Codex | 2026-02-16 |
---

## 🔍 Detalhamento de Atividades (Cards)

### ⚪ ISSUE-021 — Auditoria Total (Sênior)
> **Status:** ⏳ Planejada | **Responsável:** Codex | **Data:** 2026-02-17
> - 📄 [Ver relatório completo](./prompts/relatorios/ISSUE-021-auditoria-total-repositorio-analise-senior.md)
> - 🛠️ [Ver execução técnica](./prompts/execucoes/EXEC-021-auditoria-total-repositorio-analise-senior.md)

### 🟢 ISSUE-023 — Estabilidade Runtime
> **Status:** ✅ Concluída | **Responsável:** Codex | **Data:** 2026-02-17
> - 📄 [Ver relatório completo](./prompts/relatorios/ISSUE-023-auditoria-corretiva-estabilidade-runtime.md)
> - 📦 [Ver execução](./prompts/execucoes/EXEC-023-auditoria-corretiva-estabilidade-runtime.md)

### 🟢 ISSUE-012 — Auditoria SRC
> **Status:** ✅ Concluída | **Responsável:** Codex | **Data:** 2026-02-17
> - 📄 [Ver relatório completo](./prompts/relatorios/ISSUE-012-auditoria-codigo-testes-adr35.md)

---

## ⚙️ Workflow e Sincronização

### 5. Fluxo Padrão para Novas Issues
1. Registrar issue em `execution_queue.csv`.
2. Executar `python tools/sync_hub.py` para atualizar esta página.
3. Criar próximo ADR sequencial quando necessário.

### 6. Checklist de Fechamento
- [ ] Links internos validados.
- [ ] Índices atualizados.
- [ ] Conformidade IEEE/ISO revisada.

### 7. Referências
- **ISO/IEC 12207** — Lifecycle Processes.
- **IEEE 730** — Quality Assurance.

---
<p align="center">
  <sub><b>PyFolds HUB_CONTROLE</b> • Atualizado via <code>sync_hub.py</code></sub>
</p>
