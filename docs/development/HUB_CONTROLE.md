# HUB_CONTROLE — Gestão de Issues e Conflitos de Agentes

## Objetivo
Centralizar a fila de execução de documentação e governança para evitar conflitos entre agentes e manter rastreabilidade conforme ISO/IEC 12207 e IEEE 828.

## Regras Operacionais
1. Toda issue deve referenciar uma ADR quando alterar arquitetura, processo ou padrão técnico.
2. Apenas uma issue pode ficar em estado **Em Progresso** por agente.
3. Mudanças em `/docs/governance` exigem atualização de índice (`INDEX.md`) e deste HUB.
4. Ao concluir uma issue, registrar data, responsável e artefatos alterados.

## Fila de Execução

| ID | Tema | Status | Responsável | Data | Artefatos |
| :-- | :--- | :----- | :---------- | :--- | :-------- |
| ISSUE-001 | Reestruturação sistêmica de `/docs` e raiz (governança v1.0.0) | ✅ Concluída | Codex | 2026-02-16 | `docs/governance/*`, `docs/architecture/*`, `docs/public/guides/*` |
| ISSUE-002 | Unificação e serialização da série de ADRs | ✅ Concluída | Codex | 2026-02-16 | `docs/governance/adr/*`, `docs/governance/adr/INDEX.md` |
| ISSUE-003 | Revisão final de links cruzados e documentação pública | 🟡 Planejada | A definir | - | `README.md`, `docs/README.md` |

## Fluxo Padrão para Novas Issues
1. Registrar issue na tabela acima.
2. Verificar se há ADR aplicável.
3. Criar próximo ADR sequencial (`ADR-XXX-*`) quando necessário.
4. Executar mudanças em branch dedicada.
5. Atualizar este HUB e os índices de governança.

## Referências
- ISO/IEC 12207 — Software Life Cycle Processes.
- IEEE 828 — Software Configuration Management Plans.
- IEEE 730 — Software Quality Assurance.
