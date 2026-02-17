# 🚀 HUB_CONTROLE — Centro de Governança PyFolds

<p align="center">
  <img src="pyfoldnovo.PNG" width="550" alt="PyFolds Logo Banner">
</p>

<p align="center">
  <a href="#-documentação-de-governança">
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
> **RESTRIÇÃO DE ACESSO:** Este HUB centraliza a fila de execução para o projeto **PyFolds**. Uso exclusivo para mantenedores e agentes de IA visando conformidade com a **IEEE 730**.

---

## 🗺️ Painel de Comando (Navegação UX)
*Selecione a camada de documentação desejada:*

| Camada Técnica (Engenharia) | Camada de Governança (Estratégia) |
| :--- | :--- |
| <a href="DEVELOPMENT.md"><img src="https://img.shields.io/badge/Dev_Setup-0A3069?style=for-the-badge&logo=python&logoColor=FFD700"></a> | <a href="../governance/MASTER_PLAN.md"><img src="https://img.shields.io/badge/Master_Plan-FFD700?style=for-the-badge&logo=googlesheets&logoColor=0A3069"></a> |
| <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/Workflow-0A3069?style=for-the-badge&logo=git&logoColor=FFD700"></a> | <a href="../governance/adr/INDEX.md"><img src="https://img.shields.io/badge/Decision_Log-FFD700?style=for-the-badge&logo=blueprint&logoColor=0A3069"></a> |
| <a href="release_process.md"><img src="https://img.shields.io/badge/Deploy_CI-0A3069?style=for-the-badge&logo=semantic-release&logoColor=FFD700"></a> | <a href="guides/DOC-UX-IEEE-REVIEW.md"><img src="https://img.shields.io/badge/Review_UX-FFD700?style=for-the-badge&logo=adobe-experience-manager&logoColor=0A3069"></a> |

---

## 📊 Fila de Execução Sincronizada
*Dados processados em: 17/02/2026*

| ID | Status | Tema Central | Responsável | Data |
| :--- | :--- | :--- | :--- | :--- |
| `ISSUE-021` | ⏳ **PLAN** | Auditoria total do repositório (Sênior) | Codex | 2026-02-17 |
| `ISSUE-023` | ✅ **DONE** | Auditoria de estabilidade runtime | Codex | 2026-02-17 |
| `ISSUE-012` | 🔄 **REVIEW** | Auditoria de código em `src` | Codex | 2026-02-17 |

---

## 🧩 Detalhamento de Sprints (Compliance IEEE 730)

> [!NOTE]
> ### ⚪ ISSUE-021 — Auditoria Total do Repositório
> **Status:** ⏳ Planejada | **Responsável:** Codex | **Data:** 17/02/2026
> Análise sênior completa sem execução de mudanças de produto imediatas.
> - 📄 [Ver relatório completo](./prompts/relatorios/ISSUE-021-auditoria-total-repositorio-analise-senior.md)
> - 🛠️ [Ver execução técnica](./prompts/execucoes/EXEC-021-auditoria-total-repositorio-analise-senior.md)

> [!TIP]
> ### 🟢 ISSUE-023 — Estabilidade e Consistência
> **Status:** ✅ Concluída | **Data:** 17/02/2026
> - 📄 [Relatório de Auditoria](./prompts/relatorios/ISSUE-023-auditoria-corretiva-estabilidade-runtime.md)
> - 📦 [Artefato de Execução](./prompts/execucoes/EXEC-023-auditoria-corretiva-estabilidade-runtime.md)

---

## 🔄 Protocolo de Operação (ISO/IEC 12207)

Para garantir o determinismo no desenvolvimento do **PyFolds**, siga o ciclo:

1. **Input:** Registrar demanda em `execution_queue.csv`.
2. **Sync:** Executar `python tools/sync_hub.py`.
3. **Audit:** Vincular cada mudança a uma **ADR**.

```bash
# Sincronização e validação de integridade do HUB
python tools/sync_hub.py --check
