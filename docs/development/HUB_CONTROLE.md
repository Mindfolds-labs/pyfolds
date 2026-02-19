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
| ISSUE-001 | Planejada | Implementar LTD explícita em sinapses | Codex | 2026-02-19 |
| ISSUE-002 | Planejada | Refatorar HUB com cards limpos e split view de links | Codex | 2026-02-19 |
<!-- HUB:QUEUE:END -->

### 4.1 Cards (UI limpa)

<table>
  <tr>
    <td width="68%" valign="top">

<!-- HUB:CARDS:BEGIN -->
> [!NOTE]
> <span style="display:inline-block;border:1px solid #9ec5fe;background:#f8fbff;padding:8px 12px;border-radius:8px;">**ISSUE-001** · Implementar LTD explícita em sinapses</span>
>
> **Status:** ⏳ Planejada  
> **Responsável:** Codex  
> **Data:** 2026-02-19  
> **Prioridade:** `Alta` · **Área:** `Core/Plasticidade`  
>
> <a href="./prompts/relatorios/ISSUE-001-implementar-ltd-explicita-em-sinapses.md">📄 Relatório</a> · <a href="./prompts/execucoes/EXEC-001-correcoes-ordem-neuronal.md">🛠️ Execução</a>

> [!NOTE]
> <span style="display:inline-block;border:1px solid #b7ebc6;background:#f6fff8;padding:8px 12px;border-radius:8px;">**ISSUE-002** · Refatorar HUB com cards limpos e split view de links</span>
>
> **Status:** ⏳ Planejada  
> **Responsável:** Codex  
> **Data:** 2026-02-19  
> **Prioridade:** `Alta` · **Área:** `Governança/UX Docs`  
>
> <a href="./prompts/relatorios/ISSUE-002-refatorar-hub-cards-limpos-split-view-links.md">📄 Relatório</a> · <a href="./prompts/execucoes/EXEC-002-refatorar-hub-cards-limpos-split-view-links.md">🛠️ Execução</a>
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

| ID | Tipo | Descrição | Impacto | Status | Issue de Correção |
| :-- | :-- | :-- | :-- | :-- | :-- |

> Campos operacionais obrigatórios no CSV: `arquivo_afetado` e `caminho_log`.

### 5.2 Regra de identificação e deduplicação

- **Regra de ID:** `FAIL-001`, `FAIL-002`, ... (sequencial, sem reutilização).
- **Assinatura de deduplicação:** `assinatura_erro + arquivo_afetado`.
- Se uma falha repetida chegar com a mesma assinatura no mesmo arquivo, atualizar `status`, `caminho_log` e metadados da linha existente em vez de abrir novo ID.

### 5.3 Vínculo com a fila de execução

- `execution_queue.csv` **alimenta** `failure_register.csv` durante a execução de cada ISSUE/EXEC, quando testes/checks capturam erro novo.
- `failure_register.csv` **retroalimenta** `execution_queue.csv` na revisão de planejamento: falhas em aberto geram (ou atualizam) linhas de execução com `Issue de Correção`.
- Momento de sincronização: no fechamento de cada execução e antes da atualização dos blocos `HUB:QUEUE` e `HUB:CARDS`.


## Histórico em legado

- Relatórios/ISSUEs anteriores ao marco 2.0: [`docs/development/legado/issues/`](./legado/issues/)
- Execuções/EXECs anteriores ao marco 2.0: [`docs/development/legado/execucoes/`](./legado/execucoes/)
- Relatórios consolidados de arquivamento: [`docs/development/legado/relatorios/`](./legado/relatorios/)
