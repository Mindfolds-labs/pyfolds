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
- Guia de revisão UX/IEEE: [`guides/DOC-UX-IEEE-REVIEW.md`](guides/DOC-UX-IEEE-REVIEW.md)
- Governança (raiz): [`../governance/MASTER_PLAN.md`](../governance/MASTER_PLAN.md)
- ADR index canônico: [`../governance/adr/INDEX.md`](../governance/adr/INDEX.md)

## 3. Regras Operacionais
1. Toda issue deve referenciar uma ADR quando alterar arquitetura, processo ou padrão técnico.
2. Apenas uma issue pode ficar em estado **Em Progresso** por agente.
3. Mudanças em `/docs/governance` exigem atualização de índice (`INDEX.md`) e deste HUB.
4. Ao concluir uma issue, registrar data, responsável e artefatos alterados.

## 4. Fila de Execução

A fila abaixo é gerada automaticamente a partir de `docs/development/execution_queue.csv`.

### 4.0 Tabela Resumida

<!-- HUB:QUEUE:BEGIN -->
| ID | Status | Tema | Responsável | Data |
| :-- | :-- | :-- | :-- | :-- |
| ISSUE-001 | Concluída | Reestruturação sistêmica de /docs e raiz (governança v1.0.0) | Codex | 2026-02-16 |
| ISSUE-002 | Concluída | Unificação e serialização da série de ADRs | Codex | 2026-02-16 |
| ISSUE-003 | Concluída | Auditoria completa do repositório (docs + src + .github + examples + tests) | Codex | 2026-02-16 |
| ISSUE-004 | Concluída | Consolidação do hub interno e navegação em docs/development | Codex | 2026-02-16 |
| ISSUE-005 | Concluída | Consolidação total: implementar plano de ação da auditoria (3 sprints) | Codex | 2026-02-17 |
| ISSUE-006 | Cancelada | Número reservado (não utilizado intencionalmente) | Codex | 2026-02-16 |
| ISSUE-007 | Concluída | Consolidação final do workflow e normalização total de prompts | Codex | 2026-02-16 |
| ISSUE-008 | Concluída | Melhorar workflow de prompts com ciclo Criar-Analisar-Executar-Finalizar | Codex | 2026-02-17 |
| ISSUE-009 | Concluída | Padronização de formatos de ISSUEs para interação com IA | Codex | 2026-02-16 |
| ISSUE-010 | Concluída | Consolidação final: fechamento das ISSUEs 001-009 e limpeza documental | Codex | 2026-02-16 |
| ISSUE-010-ESPECIAL | Concluída | Corrigir estrutura docs/ - remover soltos e órfãos | Codex | 2026-02-17 |
| ISSUE-011 | Concluída | Consolidação de fluxo operacional e correção de cards/links | Codex | 2026-02-17 |
| ISSUE-011-ESPECIAL | Concluída | Consolidação de fluxo operacional e correção de cards/links | Codex | 2026-02-17 |
| ISSUE-012 | DONE (aguardando revisão humana) | Auditoria de código em src + testes + ADR-035 | Codex | 2026-02-17 |
| ISSUE-013 | DONE (aguardando revisão humana) | Estabilizar instalação editável em rede restrita e consolidar falhas da auditoria ADR-035 | Codex | 2026-02-17 |
<!-- HUB:QUEUE:END -->

### 4.1 ISSUE-001
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-001** — Reestruturação Sistêmica de /docs e Raiz  
*Governança v1.0.0*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-16

📄 [Ver relatório completo](./prompts/relatorios/README.md)

</div>

### 4.2 ISSUE-002
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-002** — Unificação e Serialização da Série de ADRs  
*Governança / ADR*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-16

📄 [Ver relatório completo](./prompts/relatorios/README.md)

</div>

### 4.3 ISSUE-003
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-003** — Auditoria Completa do Repositório  
*Documentação / Governança*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-16

📄 [Ver relatório completo](./prompts/relatorios/ISSUE-003-auditoria-completa.md)

</div>

### 4.4 ISSUE-004
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-004** — Consolidação do Hub Interno  
*Desenvolvimento / Processo*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-16

📄 [Ver relatório completo](./prompts/relatorios/README.md)

</div>

### 4.5 ISSUE-005
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-005** — Consolidação total: implementar plano de ação da auditoria (3 sprints)  
*Documentação / Multi-área*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-17

📄 [Ver relatório completo](./prompts/relatorios/ISSUE-005-plano-acao-consolidacao.md)

</div>

### 4.6 ISSUE-007
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-007** — Consolidação final do workflow e normalização total de prompts  
*Documentação / Processo*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-16

📄 [Ver relatório completo](./prompts/relatorios/ISSUE-007-consolidacao-final.md)

</div>

### 4.7 ISSUE-008
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-008** — Melhorar workflow de prompts com ciclo Criar-Analisar-Executar-Finalizar  
*Documentação / Processo*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-17

📄 [Ver relatório completo](./prompts/relatorios/ISSUE-008-melhoria-workflow-prompts.md)

</div>

### 4.8 ISSUE-009
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-009** — Padronização de formatos de ISSUEs para interação com IA  
*Documentação / Processo*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-16

📄 [Ver relatório completo](./prompts/relatorios/ISSUE-009-padronizacao-formatos-ia.md)

</div>

### 4.9 ISSUE-010
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-010** — Consolidação final: fechamento das ISSUEs 001-009 e limpeza documental  
*Governança / Documentação*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-16

📄 [Ver relatório completo](./prompts/relatorios/ISSUE-010-consolidacao-final-fechar-tudo.md)

</div>

### 4.10 ISSUE-010-ESPECIAL
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-010-ESPECIAL** — Corrigir estrutura docs/ - remover soltos e órfãos  
*Governança / Documentação*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-17

📄 [Ver relatório completo](./prompts/relatorios/ISSUE-010-ESPECIAL-corrigir-estrutura-docs.md)

</div>


### 4.11 ISSUE-011
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-011** — Consolidação de fluxo operacional e correção de cards/links  
*Governança / Documentação*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-17

📄 [Ver relatório completo](./prompts/relatorios/ISSUE-011-consolidacao-fluxo.md)

</div>

### 4.12 ISSUE-011-ESPECIAL
<div style="background: #d4edda; border-left: 4px solid #28a745; padding: 12px;">

**ISSUE-011-ESPECIAL** — Consolidação de fluxo operacional e correção de cards/links  
*Governança / Documentação*

Status: ✅ Concluída | Responsável: Codex | Data: 2026-02-17

📄 [Ver relatório completo](./prompts/relatorios/ISSUE-011-ESPECIAL-consolidacao-fluxo.md)

</div>

### 4.8 Padrão de Cores

- ✅ Concluída → `#d4edda` (fundo) | `#28a745` (borda esquerda)
- 🔄 Progresso → `#fff3cd` (fundo) | `#ffc107` (borda esquerda)
- ⏳ Planejada → `#e2e3e5` (fundo) | `#6c757d` (borda esquerda)
- ❌ Bloqueada → `#f8d7da` (fundo) | `#dc3545` (borda esquerda)

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

## 8. Como atualizar a fila manualmente

```bash
python tools/sync_hub.py
python tools/sync_hub.py --check
```

> Nota GitHub Actions: para o workflow de sincronização abrir PR automaticamente, habilite
> **Settings > Actions > General > Workflow permissions > Allow GitHub Actions to create and approve pull requests**.
