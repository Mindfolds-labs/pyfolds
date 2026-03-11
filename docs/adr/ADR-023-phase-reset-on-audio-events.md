# Context
Deriva de fase pode degradar sincronização em fluxos longos.

# Decision
Adicionar `reset_phase_if_event` e acioná-lo opcionalmente via eventos do envelope.

# Rationale
Fornece alinhamento temporal por eventos relevantes.

# Tradeoffs
Pode introduzir resets excessivos em sinais muito abruptos.

# Scientific references
Lakatos et al. (2008); Giraud & Poeppel (2012).
