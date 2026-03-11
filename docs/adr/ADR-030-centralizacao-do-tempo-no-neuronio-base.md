# Context
Há múltiplas fontes de verdade para tempo (`time_counter` em mixins e `global_time_ms` no core), o que aumenta risco de inconsistência científica.

# Decision
Centralizar o avanço temporal no neurônio base como autoridade única, mantendo `time_counter` como alias compatível de API.

# Rationale
Evita incrementos duplicados, simplifica raciocínio de execução e melhora reprodutibilidade entre composições de mixins.

# Tradeoffs
Migração exige cuidado com checkpoints legados e testes de regressão temporal.

# Consequences
Mixins deixam de ser donos do relógio e passam a consumir o tempo fornecido pelo core.
