# 🚀 HUB_CONTROLE — Gestão de Issues e Governança

<p align="center">
  <img src="pyfoldnovo.PNG" width="250" alt="Logo PyFolds">
</p>

<p align="center">
  <a href="#id-do-documento">
    <img src="https://img.shields.io/badge/ID-DEV--HUB--CTRL--001-0A3069?style=for-the-badge&logo=target" alt="ID">
  </a>
  <a href="../../actions">
    <img src="https://img.shields.io/badge/Status-Ativo-FFD700?style=for-the-badge&logo=github-actions&logoColor=0A3069" alt="Status">
  </a>
  <a href="#-referências">
    <img src="https://img.shields.io/badge/Normas-IEEE_|_ISO-0A3069?style=for-the-badge&logo=bookstack" alt="Normas">
  </a>
</p>

> [!IMPORTANT]
> **Atenção:** Este HUB é uma ferramenta de governança interna para o projeto **PyFolds**. Não deve ser utilizado por usuários finais, apenas por mantenedores e agentes de IA.

---

## 🗺️ Painel de Navegação Técnica
*Acesse os artefatos de engenharia clicando nos botões abaixo:*

<p align="left">
  <a href="DEVELOPMENT.md">
    <img src="https://img.shields.io/badge/Engenharia-DEVELOPMENT.md-0A3069?style=for-the-badge&logo=python&logoColor=FFD700" alt="Dev">
  </a>
  <a href="CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/Contribuição-CONTRIBUTING.md-0A3069?style=for-the-badge&logo=github&logoColor=FFD700" alt="Contrib">
  </a>
  <a href="release_process.md">
    <img src="https://img.shields.io/badge/Release-Process-0A3069?style=for-the-badge&logo=semantic-release&logoColor=FFD700" alt="Release">
  </a>
</p>

<p align="left">
  <a href="../governance/MASTER_PLAN.md">
    <img src="https://img.shields.io/badge/Governança-MASTER_PLAN-FFD700?style=for-the-badge&logo=googlesheets&logoColor=0A3069" alt="Gov">
  </a>
  <a href="../governance/adr/INDEX.md">
    <img src="https://img.shields.io/badge/Decisões-ADR_INDEX-FFD700?style=for-the-badge&logo=blueprint&logoColor=0A3069" alt="ADR">
  </a>
</p>

---

## 📊 Fila de Execução

| ID | Status | Tema | Responsável | Data |
| :-- | :-- | :-- | :-- | :-- |
| `023` | ✅ Concluída | Auditoria de estabilidade runtime | Codex | 2026-02-17 |
| `021` | ⏳ Planejada | Auditoria total do repositório | Codex | 2026-02-17 |
| `012` | 🔄 Em Revisão | Auditoria de código em src | Codex | 2026-02-17 |

---

## 🧩 Timeline de Issues (Detalhamento IEEE 730)

> [!TIP]
> ### ISSUE-023 — Estabilidade Runtime
> **Status:** ✅ Concluída | **Data:** 17/02/2026
> Focada em consistência cross-módulo e resolução de falhas críticas de importação.
> - 📄 [Ver Relatório](./prompts/relatorios/ISSUE-023.md)
> - 📦 [Ver Execução](./prompts/execucoes/EXEC-023.md)

> [!WARNING]
> ### ISSUE-012 — Auditoria de Código
> **Status:** 🔄 DONE (Aguardando Revisão Humana) | **Data:** 17/02/2026
> Verificação completa da suíte de testes conforme **ADR-035**.
> - 📄 [Ver Relatório](./prompts/relatorios/ISSUE-012.md)

---

## 🔄 Fluxo de Trabalho (Engenharia de Software)

Conforme **ISO/IEC 12207**, siga o ciclo Criar-Analisar-Executar-Finalizar:

1. **Registrar:** Issue em `execution_queue.csv`.
2. **Sincronizar:** `python tools/sync_hub.py`.
3. **Desenvolver:** Branch isolada vinculada à **ADR**.
4. **Validar:** Check de integridade e revisão documental.

```bash
# Sincronização e Auditoria via CLI
python tools/sync_hub.py --check
