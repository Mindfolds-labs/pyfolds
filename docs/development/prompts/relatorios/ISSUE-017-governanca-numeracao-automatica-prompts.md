# RELATÓRIO DE CONSOLIDAÇÃO — ISSUE-017
## Governança de numeração automática e entrega completa de ISSUE/EXEC

| Metadados | |
|-----------|-|
| **Data** | 2026-02-17 |
| **Responsável** | Codex |
| **Issue** | ISSUE-017 |
| **Tipo** | GOVERNANCE |
| **Status** | ✅ Concluída |
| **Normas de Referência** | IEEE 828, IEEE 730, ISO/IEC 12207 |

---

## 1. Sumário Executivo
Foi identificado desvio no fluxo de prompts: criação de issue com numeração incorreta e sem garantir entrega conjunta de relatório + execução + CSV/HUB.

A correção define regra explícita para IA sempre calcular o próximo `ISSUE-NNN` diretamente do `execution_queue.csv`, e exige entrega completa com `EXEC-[NNN]` e sincronização do HUB.

---

## 2. Diagnóstico e Análise
### 2.1 Problema observado
- Numeração inconsistente (uso de ISSUE antiga em vez do próximo ID).
- Entrega parcial sem formalização completa de execução.

### 2.2 Causa-raiz
- Ausência de instrução operacional explícita e unificada nos guias de prompts.

### 2.3 Ação aplicada
- Atualização dos guias para tornar obrigatória a descoberta do próximo ID no CSV.
- Criação desta ISSUE-017 e do EXEC-017 para rastreabilidade completa.
- Normalização de status das ISSUE-012 a ISSUE-015 para `Concluída` na fila oficial.

---

## 3. Artefatos Atualizados
- `docs/development/prompts/README.md`
- `docs/development/prompts/relatorios/README.md`
- `docs/development/guides/ISSUE-FORMAT-GUIDE.md`
- `docs/development/prompts/relatorios/ISSUE-017-governanca-numeracao-automatica-prompts.md`
- `docs/development/prompts/execucoes/EXEC-017-governanca-numeracao-automatica-prompts.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`

---

## 4. Execução Técnica
1. Revisão dos guias de criação/execução de ISSUE.
2. Inclusão da regra formal de cálculo do próximo ID no CSV.
3. Criação de ISSUE-017 e EXEC-017 com rastreabilidade.
4. Atualização de status ISSUE-012..015 para `Concluída`.
5. Sincronização do HUB.

---

## 5. Riscos, Restrições e Mitigações
- **Risco:** agentes ignorarem o CSV e criarem IDs fora de sequência.
  - **Mitigação:** regra explícita em três documentos de governança.
- **Risco:** divergência entre CSV e HUB.
  - **Mitigação:** validação obrigatória com `tools/sync_hub.py --check`.

---

## 6. Critérios de Aceite e Status
- [x] Regra de numeração automática documentada.
- [x] Regra de entrega completa (ISSUE + EXEC + CSV + HUB) documentada.
- [x] ISSUE-017 registrada no CSV.
- [x] ISSUE-012, ISSUE-013, ISSUE-014 e ISSUE-015 atualizadas para `Concluída`.
- [x] HUB sincronizado sem divergências.

**Status final da ISSUE-017:** ✅ **Concluída**.
