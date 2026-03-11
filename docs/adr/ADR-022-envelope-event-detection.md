# Context
Eventos de borda acústica podem explicar parte do entrainment neural.

# Decision
Adicionar `detect_envelope_events` baseado em derivada retificada + threshold adaptativo.

# Rationale
Implementação simples, rápida e robusta para integração com reset/gating.

# Tradeoffs
Sensível ao parâmetro de threshold e a ruído.

# Scientific references
Oganian & Chang (2019); Doelling et al. (2014).
