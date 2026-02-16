# ISSUE-002 — Fronteira entre core sináptico e STP

## Problema
`MPJRDDendrite` assumia estados `u` e `R` dentro de `MPJRDSynapse`, mas o core sináptico expõe apenas `N`, `I`, `protection`, `sat_time` e `eligibility`.

## Consequência
Falha de execução ao construir cache de estados em caminhos que dependem de `self.W`/forward.

## Correção proposta
- Tornar `u`/`R` opcionais no cache;
- Explicitar que STP pertence ao `ShortTermDynamicsMixin`.

## Status
✅ Corrigido no core com validação condicional.
