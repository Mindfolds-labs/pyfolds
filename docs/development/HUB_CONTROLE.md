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
  <a href="#-referências-normativas">
    <img src="https://img.shields.io/badge/STANDARDS-IEEE_|_ISO-0A3069?style=for-the-badge&logo=bookstack" alt="Normas">
  </a>
</p>

> [!IMPORTANT]
> **GOVERNANÇA AUTOMATIZADA:** Esta fila de execução é sincronizada dinamicamente a partir de `docs/development/execution_queue.csv`. Toda alteração deve seguir os protocolos de rastreabilidade das normas **IEEE 730** e **ISO/IEC 12207**.

---

## 🗺️ Painel de Comando (Navegação UX)
*Selecione a camada de documentação técnica clicando nos botões abaixo:*

| Camada de Engenharia (Dev) | Camada de Governança (Estratégia) |
| :--- | :--- |
| <a href="DEVELOPMENT.md"><img src="https://img.shields.io/badge/Dev_Setup-0A3069?style=for-the-badge&logo=python&logoColor=FFD700"></a> | <a href="../governance/MASTER_PLAN.md"><img src="https://img.shields.io/badge/Master_Plan-FFD700?style=for-the-badge&logo=googlesheets&logoColor=0A3069"></a> |
| <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/Workflow-0A3069?style=for-the-badge&logo=git&logoColor=FFD700"></a> | <a href="../governance/adr/INDEX.md"><img src="https://img.shields.io/badge/Decision_Log-FFD700?style=for-the-badge&logo=blueprint&logoColor=0A3069"></a> |
| <a href="release_process.md"><img src="https://img.shields.io/badge/Deploy_CI-0A3069?style=for-the-badge&logo=semantic-release&logoColor=FFD700"></a> | <a href="guides/DOC-UX-IEEE-REVIEW.md"><img src="https://img.shields.io/badge/Review_UX-FFD700?style=for-the-badge&logo=adobe-experience-manager&logoColor=0A3069"></a> |

---

## 📊 4.0 Tabela Resumida (Fila de Execução)

| ID | Status | Tema | Responsável | Data |
| :-- | :-- | :-- | :-- | :-- |
| ISSUE-021 | ⏳ Planejada | Auditoria total do repositório com análise sênior (sem execução de mudanças de produto) | Codex | 2026-02-17 |
| ISSUE-023 | ✅ Concluída | Auditoria corretiva de estabilidade runtime e consistência cross-módulo | Codex | 2026-02-17 |
| ISSUE-020 | ✅ Concluída | Relatório CI Docs Hub e correções para Sphinx/MyST | Codex | 2026-02-17 |
| ISSUE-019 | ✅ Concluída | Determinismo de relatórios e logs no workflow de prompts | Codex | 2026-02-17 |
| ISSUE-012 | ✅ Concluída | Auditoria de código em src + testes + ADR-035 | Codex | 2026-02-17 |
| ISSUE-010-ESPECIAL | ✅ Concluída | Corrigir estrutura docs/ - remover soltos e órfãos | Codex | 2026-02-17 |
| ISSUE-001 | ✅ Concluída | Reestruturação sistêmica de /docs e raiz (governança v1.0.0) | Codex | 2026-02-16 |
---

## 🔍 Detalhamento e Rastreabilidade (Deep Dive)

Abaixo, os detalhes extraídos dos artefatos técnicos de auditoria e execução.

### ⚪ ISSUE-021 — Auditoria Total (Análise Sênior)
* **Status:** ⏳ Planejada
* **Foco:** Análise de arquitetura e consistência sem alteração de produto.
* **Documentação:**
    * 📄 [Relatório de Auditoria](./prompts/relatorios/ISSUE-021-auditoria-total-repositorio-analise-senior.md)
    * 🛠️ [Plano de Execução Técnica](./prompts/execucoes/EXEC-021-auditoria-total-repositorio-analise-senior.md)

### 🟢 ISSUE-023 — Estabilidade Runtime
* **Status:** ✅ Concluída
* **Foco:** Correção de bugs de importação e consistência entre módulos.
* **Documentação:**
    * 📄 [Relatório de Estabilidade](./prompts/relatorios/ISSUE-023-auditoria-corretiva-estabilidade-runtime.md)
    * 📦 [Logs de Execução Técnica](./prompts/execucoes/EXEC-023-auditoria-corretiva-estabilidade-runtime.md)

---

## 🔄 Protocolo Operacional (Governança)

1. **Input:** Registrar demanda em `docs/development/execution_queue.csv`.
2. **Sync:** Executar `python tools/sync_hub.py` para atualizar este dashboard.
3. **Traceability:** Cada issue deve possuir um par Relatório/Execução em `prompts/`.

```bash
# Sincronização automática
python tools/sync_hub.py --check
