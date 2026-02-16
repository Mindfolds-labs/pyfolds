# ADR-008 — Controle homeostático com estratégia anti-windup

## Status
Proposed

## Contexto
Controladores homeostáticos acumulativos podem sofrer *windup* quando a atuação satura, gerando oscilação e recuperação lenta.

## Decisão
Adotar estratégia anti-windup no controle de homeostase:
- limitar termo integrador;
- reduzir acúmulo quando saturado;
- garantir continuidade na transição para regime nominal.

## Consequências
- menor risco de oscilação persistente;
- necessidade de ajuste adicional de ganhos e limites.

## Dependências
- Alinhado com estabilidade numérica de [ADR-006](./ADR-006-safe-weight-law.md).
- Observabilidade apoiada por [ADR-007](./ADR-007-monitoramento-de-invariantes.md).
