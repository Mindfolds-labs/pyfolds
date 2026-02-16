# ADR-006 — Invariantes e estabilidade numérica como gate de qualidade

## Status
Accepted

## Contexto
O plano técnico identifica risco de `NaN/Inf` e deriva de parâmetros em execuções longas, afetando confiabilidade científica.

## Decisão
Elevar estabilidade numérica a requisito de engenharia:

- aplicar leis seguras de atualização (clamp e validações);
- monitorar invariantes de estado (`N`, `I`, `theta`, limites de homeostase);
- incluir testes dedicados de limites e monotonicidade.

## Consequências
### Positivas
- Menor risco de estados inválidos em produção.
- Maior confiança em comparações experimentais.

### Trade-offs
- Pequeno custo de CPU por validações adicionais.
