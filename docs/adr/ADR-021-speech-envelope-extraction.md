# Context
Precisamos suportar extração de envelope com alternativas metodológicas para estudos de speech tracking.

# Decision
Implementar `extract_speech_envelope` com métodos `hilbert` e `gammatone` aproximado.

# Rationale
Permite comparação científica sem dependências pesadas adicionais.

# Tradeoffs
Gammatone simplificado é menos fiel que implementações especializadas.

# Scientific references
Ding & Simon (2014); Gross et al. (2013).
