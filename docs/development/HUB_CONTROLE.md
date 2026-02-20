# HUB_CONTROLE — Fila Ativa de Execução

> **Status:** Ativo
> **Fonte da fila:** [`execution_queue.csv`](./execution_queue.csv)

## Fila ativa

## 2. Escopo e Navegação
Este HUB **não é documentação de usuário final**. Ele deve ser usado apenas por quem mantém a base documental e os artefatos de governança.

- Índice interno de desenvolvimento: [`DEVELOPMENT.md`](DEVELOPMENT.md)
- Processo de contribuição: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Processo de release: [`release_process.md`](release_process.md)
- Prevenção de conflitos Git (canônico): [`../governance/GIT_CONFLICT_PREVENTION.md`](../governance/GIT_CONFLICT_PREVENTION.md)
- Guia de revisão UX/IEEE: [`guides/DOC-UX-IEEE-REVIEW.md`](guides/DOC-UX-IEEE-REVIEW.md)
- Governança (raiz): [`../governance/MASTER_PLAN.md`](../governance/MASTER_PLAN.md)
- ADR index canônico: [`../governance/adr/INDEX.md`](../governance/adr/INDEX.md)

## 3. Regras Operacionais
1. Toda issue deve referenciar uma ADR quando alterar arquitetura, processo ou padrão técnico.
2. Apenas uma issue pode ficar em estado **Em Progresso** por agente.
3. Mudanças em `/docs/governance` exigem atualização de índice (`INDEX.md`) e deste HUB.
4. Ao concluir uma issue, registrar data, responsável e artefatos alterados.

## 4. Fila de Execução

## Dashboard KPIs
- Dashboard HTML: [`generated/dashboard.html`](generated/dashboard.html)
- Métricas JSON: [`generated/metrics.json`](generated/metrics.json)

## Fila Próximas
Fonte: [`execution_queue.csv`](execution_queue.csv).

A fila abaixo é gerada automaticamente a partir de `docs/development/execution_queue.csv`.

### 4.0 Tabela Resumida

<!-- HUB:QUEUE:BEGIN -->
| ID | Status | Tema | Responsável | Data |
| :-- | :-- | :-- | :-- | :-- |
| ISSUE-001 | Concluída | Implementar LTD explícita em sinapses | Codex | 2026-02-19 |
| ISSUE-002 | Concluída | Refatorar HUB com cards limpos e split view de links | Codex | 2026-02-19 |
| ISSUE-003 | Concluída | Plano 1: Reposicionar issues no fluxo oficial do HUB | Codex | 2026-02-19 |
| ISSUE-004 | Concluída | Plano 2: Sanear links e navegação documental | Codex | 2026-02-19 |
| ISSUE-006 | Concluída | Plano 4: Consolidar correções de código e testes mínimos | Codex | 2026-02-19 |
| ISSUE-010 | Concluída | Falhas regressivas na suíte completa de testes | Codex | 2026-02-20 |
| ISSUE-011 | Concluída | Micro-otimização do forward e análise dos testes pulados | Codex | 2026-02-20 |
| ISSUE-012 | Concluída | Validar assinatura digital opcional e medir overhead de telemetria | Codex | 2026-02-20 |
| ISSUE-013 | Concluída | MindControl - Cérebro Externo e Mutação em Tempo Real | Codex | 2026-02-20 |
| ISSUE-014 | Concluída | Hardening de integridade de pesos (sanity check VRAM) | Codex | 2026-02-20 |
| ISSUE-014 | Concluída | Sanity check periódico de integridade de pesos e atualização docs v2.0.2 | Codex | 2026-02-20 |
| ISSUE-015 | Concluída | Hardening final do Core v2.0.3: integridade runtime, telemetria bufferizada e carga segura | Codex | 2026-02-20 |

<!-- HUB:QUEUE:END -->

### 4.1 Cards (UI limpa)

<table>
  <tr>
    <td width="68%" valign="top">

<!-- HUB:CARDS:BEGIN -->
> [!TIP]
> **ISSUE-001** · Implementar LTD explícita em sinapses
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-19  
> **Prioridade:** `Alta` · **Área:** `Core/Plasticidade`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-001-implementar-ltd-explicita-em-sinapses.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-001-correcoes-ordem-neuronal.md)

> [!TIP]
> **ISSUE-002** · Refatorar HUB com cards limpos e split view de links
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-19  
> **Prioridade:** `Alta` · **Área:** `Governança/UX Docs`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-002-refatorar-hub-cards-limpos-split-view-links.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-002-refatorar-hub-cards-limpos-split-view-links.md)

> [!TIP]
> **ISSUE-003** · Plano 1: Reposicionar issues no fluxo oficial do HUB
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-19  
> **Prioridade:** `Alta` · **Área:** `Governança/Fluxo`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-003-plano-1-reposicionar-issues-no-hub.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-003-plano-1-reposicionar-issues-no-hub.md)

> [!TIP]
> **ISSUE-004** · Plano 2: Sanear links e navegação documental
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-19  
> **Prioridade:** `Alta` · **Área:** `Documentação/Qualidade`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-004-plano-2-sanear-links-documentacao.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-004-plano-2-sanear-links-documentacao.md)

> [!TIP]
> **ISSUE-006** · Plano 4: Consolidar correções de código e testes mínimos
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-19  
> **Prioridade:** `Alta` · **Área:** `Core/Testes/Docs`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-006-plano-4-consolidar-correcoes-codigo-testes.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-006-plano-4-consolidar-correcoes-codigo-testes.md)

> [!TIP]
> **ISSUE-010** · Falhas regressivas na suíte completa de testes
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-20  
> **Prioridade:** `Alta` · **Área:** `Qualidade/Testes`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-010-falhas-regressivas-na-su-te-completa-de-testes.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-010-falhas-regressivas-na-suite-completa-de-testes.md)

> [!TIP]
> **ISSUE-011** · Micro-otimização do forward e análise dos testes pulados
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-20  
> **Prioridade:** `Alta` · **Área:** `Qualidade/Performance`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-011-otimizacao-forward-e-analise-de-skips.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-011-otimizacao-forward-e-analise-de-skips.md)

> [!TIP]
> **ISSUE-012** · Validar assinatura digital opcional e medir overhead de telemetria
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-20  
> **Prioridade:** `Alta` · **Área:** `Segurança/Performance`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-012-validacao-criptografia-e-telemetria.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-012-validacao-criptografia-e-telemetria.md)

> [!TIP]
> **ISSUE-013** · MindControl - Cérebro Externo e Mutação em Tempo Real
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-20  
> **Prioridade:** `Alta` · **Área:** `Core/Telemetria/Neuromodulação`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-013-mindcontrol.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-013-mindcontrol.md)

> [!TIP]
> **ISSUE-014** · Hardening de integridade de pesos (sanity check VRAM)
> **ISSUE-014** · Sanity check periódico de integridade de pesos e atualização docs v2.0.2
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-20  
> **Prioridade:** `Alta` · **Área:** `Segurança/Runtime`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-014-hardening-integridade-pesos-vram.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-014-hardening-integridade-pesos-vram.md)
> **Prioridade:** `Alta` · **Área:** `Segurança/Runtime/Docs`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-014-sanity-check-integridade-pesos-e-atualizacao-docs-v2-0-2.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-014-sanity-check-integridade-pesos-e-atualizacao-docs-v2-0-2.md)

> [!TIP]
> **ISSUE-015** · Hardening final do Core v2.0.3: integridade runtime, telemetria bufferizada e carga segura
>
> **Status:** ✅ Concluída  
> **Responsável:** Codex  
> **Data:** 2026-02-20  
> **Prioridade:** `Alta` · **Área:** `Segurança/Runtime/Telemetria`  
>
> 📄 [Relatório](./prompts/relatorios/ISSUE-015-hardening-core-runtime-v2-0-3.md) · 🛠️ [Execução](./prompts/execucoes/EXEC-015-hardening-core-runtime-v2-0-3.md)

<!-- HUB:CARDS:END -->

   </td>
   <td width="32%" valign="top">

#### Links rápidos

<a href="./ISSUES_SPLIT_VIEW.md">🔀 Abrir página de Split View</a><br>
<a href="./execution_queue.md">📋 Ver fila detalhada</a><br>
<a href="./failure_register.csv">🧯 Ver registro de falhas</a>

   </td>
  </tr>
</table>

> Os cards de ISSUE históricos foram removidos deste HUB e permanecem em `./legado/`.

## 5. Falhas Detectadas

Fonte canônica: [`failure_register.csv`](./failure_register.csv).

### 5.1 Esquema oficial de colunas

| ID | Tipo | Descrição | Impacto | Status de Falha | Cobertura de Teste | Issue de Correção | Teste de Regressão |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |

<!-- HUB:FAILURES:BEGIN -->
| ID | Tipo | Descrição | Impacto | Status | Cobertura | Issue de Correção | Teste | Data |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| - | - | - | - | - | - | - | - | - |
<!-- HUB:FAILURES:END -->

> Campos operacionais obrigatórios no CSV: `arquivo_afetado`, `caminho_log`, `status_cobertura`, `teste_regressao` e `evidencia_regressao`.

> Vocabulário oficial de `status_cobertura`: `aberta`, `em_correcao`, `coberta`, `validada`.

### 5.2 Regra de identificação e deduplicação

- **Regra de ID:** `FAIL-001`, `FAIL-002`, ... (sequencial, sem reutilização).
- **Assinatura de deduplicação:** `assinatura_erro + arquivo_afetado`.
- Se uma falha repetida chegar com a mesma assinatura no mesmo arquivo, atualizar `status`, `caminho_log` e metadados da linha existente em vez de abrir novo ID.

### 5.3 Vínculo com a fila de execução

- `execution_queue.csv` **alimenta** `failure_register.csv` durante a execução de cada ISSUE/EXEC, quando testes/checks capturam erro novo.
- `failure_register.csv` **retroalimenta** `execution_queue.csv` na revisão de planejamento: falhas em aberto geram (ou atualizam) linhas de execução com `Issue de Correção`.
- Momento de sincronização: no fechamento de cada execução e antes da atualização dos blocos `HUB:QUEUE` e `HUB:CARDS`.
- Política de fechamento: uma falha só pode sair de `aberta/em_correcao` para `coberta/validada` (ou status de issue encerrado) quando `teste_regressao` e `evidencia_regressao` estiverem preenchidos com referência explícita ao mesmo `FAIL-XXX`.


## Histórico em legado

- Relatórios/ISSUEs anteriores ao marco 2.0: [`docs/development/legado/issues/`](./legado/issues/)
- Execuções/EXECs anteriores ao marco 2.0: [`docs/development/legado/execucoes/`](./legado/execucoes/)
- Relatórios consolidados de arquivamento: [`docs/development/legado/relatorios/`](./legado/relatorios/)
