# Context
Mecanismos experimentais e core estável ainda compartilham pontos de integração com acoplamento alto.

# Decision
Separar formalmente o núcleo estável (`core`) dos mecanismos experimentais (`advanced/experimental` e correlatos), preservando a API pública.

# Rationale
Reduz risco de regressão no caminho crítico e facilita governança de compatibilidade.

# Tradeoffs
Aumento inicial de refatoração estrutural e manutenção de adaptadores de compatibilidade.

# Consequences
Cada mecanismo passa a declarar status (stable/experimental), contrato, métricas e cobertura mínima de testes.
