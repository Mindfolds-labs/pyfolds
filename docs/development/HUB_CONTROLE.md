# HUB_CONTROLE — Gestão de Issues e Conflitos de Agentes

> **ID do Documento:** DEV-HUB-CTRL-001  
> **Status:** Ativo  
> **Escopo:** Documentação interna de desenvolvimento e governança  
> **Normas de Referência:** ISO/IEC 12207, IEEE 828, IEEE 730

## 1. Objetivo
Centralizar a fila de execução de documentação e governança para evitar conflitos entre agentes e manter rastreabilidade.

## 2. Escopo e Navegação
Este HUB **não é documentação de usuário final**. Ele deve ser usado apenas por quem mantém a base documental e os artefatos de governança.

- Índice interno de desenvolvimento: [`DEVELOPMENT.md`](DEVELOPMENT.md)
- Processo de contribuição: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Processo de release: [`release_process.md`](release_process.md)
- Governança (raiz): [`../governance/MASTER_PLAN.md`](../governance/MASTER_PLAN.md)
- ADR index canônico: [`../governance/adr/INDEX.md`](../governance/adr/INDEX.md)

## 3. Regras Operacionais
1. Toda issue deve referenciar uma ADR quando alterar arquitetura, processo ou padrão técnico.
2. Apenas uma issue pode ficar em estado **Em Progresso** por agente.
3. Mudanças em `/docs/governance` exigem atualização de índice (`INDEX.md`) e deste HUB.
4. Ao concluir uma issue, registrar data, responsável e artefatos alterados.

## 4. Fila de Execução

| ID | Tema | Status | Responsável | Data | Artefatos |
| :-- | :--- | :----- | :---------- | :--- | :-------- |
| ISSUE-001 | Reestruturação sistêmica de `/docs` e raiz (governança v1.0.0) | ✅ Concluída | Codex | 2026-02-16 | `docs/governance/*`, `docs/architecture/*`, `docs/public/guides/*` |
| ISSUE-002 | Unificação e serialização da série de ADRs | ✅ Concluída | Codex | 2026-02-16 | `docs/governance/adr/*`, `docs/governance/adr/INDEX.md` |
| ISSUE-003 | Revisão final de links cruzados e documentação pública | 🟡 Planejada | A definir | - | `README.md`, `docs/README.md` |
| ISSUE-004 | Consolidação do hub interno e navegação em `docs/development` | ✅ Concluída | Codex | 2026-02-16 | `docs/development/HUB_CONTROLE.md`, `docs/README.md`, `docs/index.md` |

## 5. Fluxo Padrão para Novas Issues
1. Registrar issue na tabela acima.
2. Verificar se há ADR aplicável.
3. Criar próximo ADR sequencial (`ADR-XXX-*`) quando necessário.
4. Executar mudanças em branch dedicada.
5. Atualizar este HUB e os índices de governança.

## 6. Checklist de Fechamento
- [ ] Links internos validados.
- [ ] Índices atualizados (`docs/index.md`, `docs/README.md`, `docs/governance/adr/INDEX.md` quando aplicável).
- [ ] Rastreabilidade de artefatos atualizada na tabela.
- [ ] Conformidade com diretrizes IEEE/ISO revisada.

## 7. Referências
- ISO/IEC 12207 — Software Life Cycle Processes.
- IEEE 828 — Software Configuration Management Plans.
- IEEE 730 — Software Quality Assurance.
