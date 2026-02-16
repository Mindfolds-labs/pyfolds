# ADR-009 — Testes de propriedades matemáticas (property-based)

## Status
Proposed

## Contexto
Testes exemplo-a-exemplo não cobrem bem fronteiras de domínio, monotonicidade e robustez numérica em larga variedade de entradas.

## Decisão
Adicionar suíte property-based para:
- limites e domínio das variáveis de estado;
- monotonicidade/saturação esperada das leis de atualização;
- ausência de `NaN/Inf` sob entradas adversariais controladas.

## Consequências
- aumenta confiança matemática e regressão preventiva;
- pode elevar tempo de CI se não houver orçamento de amostragem.

## Dependências
- Valida hipóteses das decisões [ADR-006](./ADR-006-safe-weight-law.md), [ADR-007](./ADR-007-monitoramento-de-invariantes.md) e [ADR-008](./ADR-008-homeostase-com-anti-windup.md).
