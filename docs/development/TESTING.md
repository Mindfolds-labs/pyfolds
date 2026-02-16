# Estratégia de Testes

## Camadas
- Unitários: sinapse, dendrito, neurônio.
- Integração: forward + plasticidade + sono.
- Regressão: benchmarks e métricas estáveis.

## Cobertura mínima
Meta recomendada: 80%.

## Foco incremental (PRs de 1 dia)
- Priorizar módulos do core com menor cobertura (`accumulator`, `synapse`).
- Cobrir cenários nominais, limites e entradas inválidas em cada mecanismo.
- Validar estabilidade temporal (acúmulo em múltiplos steps) para evitar regressões silenciosas.
- Manter testes de integração/mixins separados para não bloquear validações do core.
