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
| ISSUE-012 | Concluída | Auditoria de código em src + testes + ADR-035 | Codex | 2026-02-17 |
| ISSUE-013 | Concluída | Estabilizar instalação editável em rede restrita e consolidar falhas da auditoria ADR-035 | Codex | 2026-02-17 |
| ISSUE-014 | Concluída | Auditoria SRC/Testes ADR-035 + gate CI docs hub com Sphinx/MyST/PyData | Codex | 2026-02-17 |
| ISSUE-015 | Concluída | Validar erros corrigidos + importacao pyfolds + suite completa + governanca | Codex | 2026-02-17 |
| ISSUE-001 | Concluída | Adicionar dependência linkify-it-py para MyST Parser na documentação | Codex | 2026-02-17 |
| ISSUE-017 | Concluída | Governança de numeração automática e entrega completa de ISSUE/EXEC | Codex | 2026-02-17 |
| ISSUE-018 | Concluída | Padronização de relatórios e obrigatoriedade de sync HUB | Codex | 2026-02-17 |
| ISSUE-019 | Concluída | Determinismo de relatórios e logs no workflow de prompts | Codex | 2026-02-17 |
| ISSUE-020 | Concluída | Relatório CI Docs Hub e correções para Sphinx/MyST | Codex | 2026-02-17 |
| ISSUE-021 | Planejada | Auditoria total do repositório com análise sênior (sem execução de mudanças de produto) | Codex | 2026-02-17 |
| ISSUE-023 | Concluída | Auditoria corretiva de estabilidade runtime e consistência cross-módulo | Codex | 2026-02-17 |
| ISSUE-022 | Concluída | Auditoria e correção do neurônio MPJRD (thread safety + validações + plasticidade) | Codex | 2026-02-17 |
| ISSUE-024 | Planejada | Revisão estética do HUB_CONTROLE com cards sincronizados por CSV | Codex | 2026-02-17 |
<!-- HUB:QUEUE:END -->

### 4.1 🔍 Detalhamento de Atividades (Cards)

<!-- HUB:CARDS:BEGIN -->
> [!TIP]
> **ISSUE-001** · Reestruturação sistêmica de /docs e raiz (governança v1.0.0)
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-16  
> **Prioridade:** `Alta` · **Área:** `Documentação/Governança`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-001-docs-dependency-linkify.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-001-fix-linkify-dependency.md)

> [!TIP]
> **ISSUE-002** · Unificação e serialização da série de ADRs
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-16  
> **Prioridade:** `Alta` · **Área:** `Governança/ADR`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-002-unificacao-e-serializacao-da-serie-de-adrs.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-002-unificacao-e-serializacao-da-serie-de-adrs.md)

> [!TIP]
> **ISSUE-003** · Auditoria completa do repositório (docs + src + .github + examples + tests)
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-16  
> **Prioridade:** `Alta` · **Área:** `Documentação/Governança`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-003-auditoria-completa.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-003-auditoria-completa-do-repositorio-docs-src-github-examples-tests.md)

> [!TIP]
> **ISSUE-004** · Consolidação do hub interno e navegação em docs/development
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-16  
> **Prioridade:** `Alta` · **Área:** `Desenvolvimento/Processo`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-004-consolidacao-do-hub-interno-e-navegacao-em-docs-development.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-004-consolidacao-do-hub-interno-e-navegacao-em-docs-development.md)

> [!TIP]
> **ISSUE-005** · Consolidação total: implementar plano de ação da auditoria (3 sprints)
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Alta` · **Área:** `all`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-005-plano-acao-consolidacao.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-005-consolidacao-total-implementar-plano-de-acao-da-auditoria-3-sprints.md)

> [!IMPORTANT]
> **ISSUE-006** · Número reservado (não utilizado intencionalmente)
>
> **Status:** ⚪ Cancelada  
> **Responsável:** Codex  
> **Data:** 2026-02-16  
> **Prioridade:** `Média` · **Área:** `Governança/Documentação`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-006-numero-reservado-nao-utilizado-intencionalmente.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-006-numero-reservado-nao-utilizado-intencionalmente.md)

> [!TIP]
> **ISSUE-007** · Consolidação final do workflow e normalização total de prompts
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-16  
> **Prioridade:** `Alta` · **Área:** `Documentação/Processo`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-007-consolidacao-final.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-007-consolidacao-final-do-workflow-e-normalizacao-total-de-prompts.md)

> [!TIP]
> **ISSUE-008** · Melhorar workflow de prompts com ciclo Criar-Analisar-Executar-Finalizar
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Alta` · **Área:** `Documentação/Processo`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-008-melhoria-workflow-prompts.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-008-melhorar-workflow-de-prompts-com-ciclo-criar-analisar-executar-finalizar.md)

> [!TIP]
> **ISSUE-009** · Padronização de formatos de ISSUEs para interação com IA
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-16  
> **Prioridade:** `Alta` · **Área:** `Documentação/Processo`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-009-padronizacao-formatos-ia.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-009-padronizacao-de-formatos-de-issues-para-interacao-com-ia.md)

> [!TIP]
> **ISSUE-010** · Consolidação final: fechamento das ISSUEs 001-009 e limpeza documental
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-16  
> **Prioridade:** `Alta` · **Área:** `Governança/Documentação`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-010-ESPECIAL-corrigir-estrutura-docs.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-010-consolidacao-final-fechamento-das-issues-001-009-e-limpeza-documental.md)

> [!TIP]
> **ISSUE-010-ESPECIAL** · Corrigir estrutura docs/ - remover soltos e órfãos
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Alta` · **Área:** `Governança/Documentação`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-010-ESPECIAL-corrigir-estrutura-docs.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-010-ESPECIAL-corrigir-estrutura-docs-remover-soltos-e-orfaos.md)

> [!TIP]
> **ISSUE-011** · Consolidação de fluxo operacional e correção de cards/links
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Governança/Documentação`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-011-ESPECIAL-consolidacao-fluxo.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-011-consolidacao-de-fluxo-operacional-e-correcao-de-cards-links.md)

> [!TIP]
> **ISSUE-011-ESPECIAL** · Consolidação de fluxo operacional e correção de cards/links
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Governança/Documentação`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-011-ESPECIAL-consolidacao-fluxo.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-011-ESPECIAL-consolidacao-de-fluxo-operacional-e-correcao-de-cards-links.md)

> [!TIP]
> **ISSUE-012** · Auditoria de código em src + testes + ADR-035
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Código/Testes/Governança`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-012-auditoria-codigo-testes-adr35.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-012-auditoria-de-codigo-em-src-testes-adr-035.md)

> [!TIP]
> **ISSUE-013** · Estabilizar instalação editável em rede restrita e consolidar falhas da auditoria ADR-035
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Código/Testes/Governança`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-013-estabilizar-install-editavel-rede-restrita.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-013-estabilizar-instalacao-editavel-em-rede-restrita-e-consolidar-falhas-da-auditoria-adr-035.md)

> [!TIP]
> **ISSUE-014** · Auditoria SRC/Testes ADR-035 + gate CI docs hub com Sphinx/MyST/PyData
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Código/Testes/Governança`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-014-auditoria-src-testes-adr35-ci-docs-hub.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-014-auditoria-src-testes-adr-035-gate-ci-docs-hub-com-sphinx-myst-pydata.md)

> [!TIP]
> **ISSUE-015** · Validar erros corrigidos + importacao pyfolds + suite completa + governanca
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Código/Testes/Governança`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-015-validar-erros-imports-testes-e-governanca.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-015-validar-erros-corrigidos-importacao-pyfolds-suite-completa-governanca.md)

> [!TIP]
> **ISSUE-001** · Adicionar dependência linkify-it-py para MyST Parser na documentação
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Alta` · **Área:** `DOCS`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-001-docs-dependency-linkify.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-001-fix-linkify-dependency.md)

> [!TIP]
> **ISSUE-017** · Governança de numeração automática e entrega completa de ISSUE/EXEC
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Alta` · **Área:** `Governança/Processo`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-017-governanca-numeracao-automatica-prompts.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-017-governanca-numeracao-automatica-prompts.md)

> [!TIP]
> **ISSUE-018** · Padronização de relatórios e obrigatoriedade de sync HUB
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Governança/Processo`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-018-padronizacao-relatorios-sync-hub-obrigatorio.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-018-padronizacao-relatorios-sync-hub-obrigatorio.md)

> [!TIP]
> **ISSUE-019** · Determinismo de relatórios e logs no workflow de prompts
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Governança/Processo`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-019-determinismo-relatorios-logs-workflow-prompts.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-019-determinismo-relatorios-logs-workflow-prompts.md)

> [!TIP]
> **ISSUE-020** · Relatório CI Docs Hub e correções para Sphinx/MyST
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Documentação/Governança`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-020-relatorio-ci-docs-hub-sphinx-myst.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-020-relatorio-ci-docs-hub-e-correcoes-para-sphinx-myst.md)

> [!NOTE]
> **ISSUE-021** · Auditoria total do repositório com análise sênior (sem execução de mudanças de produto)
>
> **Status:** ⏳ Planejada  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Governança/Processo`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-021-auditoria-total-repositorio-analise-senior.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-021-auditoria-total-repositorio-analise-senior.md)

> [!TIP]
> **ISSUE-023** · Auditoria corretiva de estabilidade runtime e consistência cross-módulo
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Código/Testes/Governança`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-023-auditoria-corretiva-estabilidade-runtime.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-023-auditoria-corretiva-estabilidade-runtime.md)

> [!TIP]
> **ISSUE-022** · Auditoria e correção do neurônio MPJRD (thread safety + validações + plasticidade)
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Crítica` · **Área:** `Código/Core`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-022-auditoria-neuron-thread-safety-plasticidade.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-022-auditoria-neuron-thread-safety-plasticidade.md)

> [!NOTE]
> **ISSUE-024** · Revisão estética do HUB_CONTROLE com cards sincronizados por CSV
>
> **Status:** ⏳ Planejada  
> **Responsável:** Codex  
> **Data:** 2026-02-17  
> **Prioridade:** `Alta` · **Área:** `Documentação/Processo`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-024-correcao-tipos-telemetria-apis.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-024-revisao-estetica-hub-controle.md)

<!-- HUB:CARDS:END -->

### 4.2 Legenda visual de status

- ✅ **Concluída**
- 🚧 **Em Progresso**
- ⏳ **Planejada**
- ❌ **Bloqueada**
- ⚪ **Cancelada**

## 5. Fluxo Padrão para Novas Issues
1. Registrar issue em `execution_queue.csv` e sincronizar automaticamente tabela e cards com `python tools/sync_hub.py`.
2. Verificar se há ADR aplicável.
3. Criar próximo ADR sequencial (`ADR-XXX-*`) quando necessário.
4. Executar mudanças em branch dedicada.
5. Confirmar os links de relatório/execução gerados no card da issue e atualizar índices de governança quando aplicável.

## 6. Checklist de Fechamento
- [ ] Links internos validados.
- [ ] Índices atualizados (`docs/index.md`, `docs/README.md`, `docs/governance/adr/INDEX.md` quando aplicável).
- [ ] Rastreabilidade de artefatos atualizada na tabela.
- [ ] Conformidade com diretrizes IEEE/ISO revisada.

## 7. Referências
- ISO/IEC 12207 — Software Life Cycle Processes.
- IEEE 828 — Software Configuration Management Plans.
- IEEE 730 — Software Quality Assurance.

## 8. Workflow e Sincronização

```bash
python tools/sync_hub.py
python tools/sync_hub.py --check
```

> O processo de sincronização atualiza simultaneamente a tabela resumida e a seção de cards usando o CSV como fonte única de verdade.

> Nota GitHub Actions: para o workflow de sincronização abrir PR automaticamente, habilite
> **Settings > Actions > General > Workflow permissions > Allow GitHub Actions to create and approve pull requests**.

