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
> **GOVERNANÇA ATIVA:** Este dashboard centraliza a fila de execução extraída de `execution_queue.csv`. O objetivo é garantir o cumprimento da norma **IEEE 730** e evitar conflitos entre agentes.

---

## 📊 4.0 Fila de Execução Sincronizada
*Dados processados em: 17/02/2026*

| ID | Status | Tema | Responsável | Data |
| :-- | :-- | :-- | :-- | :-- |
| `ISSUE-023` | ✅ Concluída | Auditoria de estabilidade runtime | Codex | 2026-02-17 |
| `ISSUE-021` | ⏳ Planejada | Auditoria total do repositório (Sênior) | Codex | 2026-02-17 |
| `ISSUE-020` | ✅ Concluída | Relatório CI Docs Hub (Sphinx/MyST) | Codex | 2026-02-17 |
| `ISSUE-012` | ✅ Concluída | Auditoria de código em src + ADR-035 | Codex | 2026-02-17 |
| `ISSUE-011` | ✅ Concluída | Consolidação de fluxo operacional | Codex | 2026-02-17 |
| `ISSUE-005` | ✅ Concluída | Implementar plano de ação da auditoria | Codex | 2026-02-17 |

---

## 🔍 4.21 Detalhamento de Rastreabilidade (Sprint Atual)

Aqui estão os artefatos técnicos vinculados às atividades em aberto ou recém-concluídas:

### ⚪ ISSUE-021 — Auditoria Total (Análise Sênior)
> **Escopo:** Análise estrutural completa sem alterações de produto imediato.
* **Status:** ⏳ Planejada.
* **Responsável:** Codex.
* **Documentação Técnica:**
    * 📄 [Ver Relatório de Auditoria](./prompts/relatorios/ISSUE-021-auditoria-total-repositorio-analise-senior.md)
    * 🛠️ [Ver Execução Técnica](./prompts/execucoes/EXEC-021-auditoria-total-repositorio-analise-senior.md)

### 🟢 ISSUE-023 — Estabilidade Runtime
> **Escopo:** Consistência cross-módulo e estabilização de runtime.
* **Status:** ✅ Concluída.
* **Artefatos:**
    * 📄 [Relatório Técnico](./prompts/relatorios/ISSUE-023-auditoria-corretiva-estabilidade-runtime.md)
    * 📦 [Execução Técnica](./prompts/execucoes/EXEC-023-auditoria-corretiva-estabilidade-runtime.md)

---

## 🔄 Protocolo de Operação (ISO/IEC 12207)

Para manter a integridade do **PyFolds**, o fluxo de governança deve seguir estas etapas:

1. **Registrar:** Adicione a demanda no arquivo `docs/development/execution_queue.csv`.
2. **Sincronizar:** Execute o comando para atualizar este HUB:
   ```bash
   python tools/sync_hub.py --check
