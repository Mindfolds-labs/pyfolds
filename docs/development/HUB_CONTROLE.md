# 🚀 HUB_CONTROLE — Gestão de Issues e Governança

<p align="left">
  <img src="https://img.shields.io/badge/ID-DEV--HUB--CTRL--001-blue?style=for-the-badge" alt="ID">
  <img src="https://img.shields.io/badge/Status-Ativo-success?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Normas-ISO%2FIEC%20%7C%20IEEE-orange?style=for-the-badge" alt="Normas">
</p>

> [!IMPORTANT]
> **Atenção:** Este HUB é uma ferramenta de governança interna. Não deve ser utilizado por usuários finais, apenas por mantenedores e agentes de IA.

---

## 🗺️ Navegação Rápida

| Documento | Função | Link |
| :--- | :--- | :--- |
| 🛠️ **Desenvolvimento** | Guia de Setup e Execução | [`DEVELOPMENT.md`](DEVELOPMENT.md) |
| 🤝 **Contribuição** | Regras de PR e Commits | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| 📜 **Master Plan** | Plano de Governança Raiz | [`MASTER_PLAN.md`](../governance/MASTER_PLAN.md) |
| ⚖️ **ADR Index** | Decisões de Arquitetura | [`ADR/INDEX.md`](../governance/adr/INDEX.md) |

---

## 📊 Fila de Execução

| ID | Status | Tema | Responsável | Data |
| :-- | :-- | :-- | :-- | :-- |
| `023` | ✅ Concluída | Auditoria de estabilidade runtime | Codex | 2026-02-17 |
| `021` | ⏳ Planejada | Auditoria total do repositório | Codex | 2026-02-17 |
| `012` | 🔄 Em Revisão | Auditoria de código em src | Codex | 2026-02-17 |

---

## 🧩 Detalhamento de Issues (Timeline)

> [!TIP]
> ### ISSUE-023 — Estabilidade Runtime
> **Status:** ✅ Concluída | **Data:** 17/02/2026
>
> Focada em consistência cross-módulo e resolução de falhas críticas de importação.
> - 📄 [Ver Relatório](./prompts/relatorios/ISSUE-023.md)
> - 📦 [Ver Execução](./prompts/execucoes/EXEC-023.md)

> [!WARNING]
> ### ISSUE-012 — Auditoria de Código
> **Status:** 🔄 DONE (Aguardando Revisão Humana) | **Data:** 17/02/2026
>
> Verificação completa da suíte de testes conforme ADR-035.
> - 📄 [Ver Relatório](./prompts/relatorios/ISSUE-012.md)

---

## 🔄 Fluxo de Trabalho (Engenharia)

Para manter a ordem e evitar conflitos entre agentes, siga rigorosamente o fluxo:

1. **Registrar:** Adicione a demanda em `execution_queue.csv`.
2. **Sincronizar:** Execute `python tools/sync_hub.py` para atualizar esta página.
3. **Executar:** Crie uma branch dedicada e vincule a uma **ADR**.
4. **Validar:** Verifique se os links estão funcionais antes do Merge.

### Automação
```bash
# Sincronização manual
python tools/sync_hub.py --check
