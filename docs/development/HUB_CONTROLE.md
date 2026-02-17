# 🚀 HUB_CONTROLE — Gestão de Issues e Conflitos

<p align="left">
  <a href="#"><img src="https://img.shields.io/badge/ID-DEV--HUB--CTRL--001-0052FF?style=for-the-badge&logo=target" alt="Doc ID"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Ativo-238636?style=for-the-badge&logo=github-actions" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/ISO%2FIEC-12207-orange?style=for-the-badge" alt="Normas"></a>
</p>

> [!NOTE]
> **Propósito:** Centralizar a fila de execução e governança para evitar conflitos de agentes e manter a rastreabilidade total do ciclo de vida do software.

---

## 🧭 Navegação de Governança

| Recurso | Descrição | Link |
| :--- | :--- | :--- |
| 🛠️ **Desenvolvimento** | Guia técnico e setup de ambiente | [`DEVELOPMENT.md`](DEVELOPMENT.md) |
| 🧪 **Contribuição** | Padrões de commits e PRs | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| 📜 **Master Plan** | Visão estratégica de governança | [`MASTER_PLAN.md`](../governance/MASTER_PLAN.md) |
| ⚖️ **Decision Log** | Índice de ADRs (Decisões de Arquitetura) | [`ADR/INDEX.md`](../governance/adr/INDEX.md) |

---

## 📊 Fila de Execução (Auto-Sync)

A tabela abaixo reflete o estado atual do repositório processado por `sync_hub.py`.

| ID | Status | Tema | Responsável | Data |
| :-- | :--- | :--- | :--- | :--- |
| `023` | ✅ | Auditoria de estabilidade runtime | Codex | 2026-02-17 |
| `021` | ⏳ | Auditoria total do repositório (Sênior) | Codex | 2026-02-17 |
| `012` | 🔄 | Auditoria de código em src + ADR-035 | Codex | 2026-02-17 |
| `001` | ✅ | Reestruturação sistêmica de /docs | Codex | 2026-02-16 |

---

## 🧩 Cards de Execução Detalhada

Aqui são detalhados os artefatos gerados em cada sprint de documentação.

### Recentes e Em Progresso

> [!IMPORTANT]
> **ISSUE-023 — Auditoria Corretiva de Estabilidade**
> - **Foco:** Consistência cross-módulo e runtime.
> - **Entrega:** Estabilização de imports e correção de falhas críticas.
> - 📄 [Relatório](./prompts/relatorios/ISSUE-023-auditoria.md) | 📦 [Execução](./prompts/execucoes/EXEC-023.md)

> [!CAUTION]
> **ISSUE-012 — Auditoria de Código (Review Requerido)**
> - **Foco:** Validação de `src` + suíte de testes vs ADR-035.
> - **Status:** DONE (Aguardando aprovação humana).
> - 📄 [Ver Relatório](./prompts/relatorios/ISSUE-012.md)

> [!TIP]
> **ISSUE-021 — Planejamento de Auditoria Sênior**
> - **Foco:** Análise de arquitetura sem alteração de produto.
> - **Status:** ⏳ Aguardando janela de execução.
> - 📄 [Draft do Plano](./prompts/relatorios/ISSUE-021.md)

---

## 🛠️ Regras Operacionais (Guidelines)

Um engenheiro deve seguir este workflow para garantir a integridade do HUB:

1. **Atomicidade:** Apenas uma issue em estado `In Progress` por agente.
2. **Rastreabilidade:** Toda alteração de arquitetura **deve** referenciar uma ADR.
3. **Sincronismo:** Mudanças em `/governance` exigem atualização imediata dos índices.
4. **Fechamento:** Registrar data, responsável e artefatos antes de marcar como `Concluída`.

---

## 💻 Comandos de Manutenção

Utilize as ferramentas internas para manter o Hub atualizado:

```bash
# Sincronizar tabela de issues com o CSV de execução
python tools/sync_hub.py

# Validar se todos os links internos estão funcionais
python tools/sync_hub.py --check
