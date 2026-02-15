# ISSUE-001 — Perda de especificidade Hebbiana no update dendrítico

## Problema
O update das sinapses usava um `pre_rate` vetorial inteiro em cada sinapse, aproximando a atualização de todas as sinapses para um mesmo valor médio.

## Consequência
- Redução de seletividade por sinapse.
- Aprendizado menos discriminativo.
- Risco de convergência para estados homogêneos.

## Correção proposta
Aplicar taxa pré-sináptica local por índice sináptico, i.e., `pre_rate[i]` para a sinapse `i`.

## Status
✅ Corrigido no core.
