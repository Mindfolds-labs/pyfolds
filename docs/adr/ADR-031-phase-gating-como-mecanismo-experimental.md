# Context
`phase_gating` agrega valor científico exploratório, mas não possui estabilidade equivalente aos mecanismos centrais.

# Decision
Classificar `phase_gating` como mecanismo experimental opt-in, com toggle explícito e telemetria própria.

# Rationale
Permite evolução rápida da hipótese sem contaminar contratos estáveis do runtime principal.

# Tradeoffs
Maior superfície de configuração e necessidade de documentação mais clara por release.

# Consequences
Resultados com `phase_gating` ligado devem ser marcados como experimentais em relatórios e benchmarks.
