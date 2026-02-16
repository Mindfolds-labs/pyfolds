# ADR-006 — Safe Weight Law (clamp + validação numérica)

## Status
Proposed

## Contexto
Atualizações de peso sem guardrails podem produzir `NaN/Inf` ou saturação não-controlada em regimes extremos de aprendizado.

## Decisão
Aplicar *safe weight law* nas atualizações:
- clamp explícito em intervalo configurável;
- validação de finitude (`isfinite`) antes e depois da atualização;
- política fail-fast para estados inválidos.

## Consequências
- melhora estabilidade numérica de longo prazo;
- pode reduzir plasticidade em cenários de forte exploração, exigindo calibração de limites.

## Dependências
- Base de validação operacional em [ADR-004](./ADR-004-validacao-multicamada.md).
