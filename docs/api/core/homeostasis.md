# API Core — `HomeostasisController`

Controla adaptação de limiar somático.

Estados principais:

- `theta`: limiar corrente
- `r_hat`: taxa média estimada

Método principal:

- `update(spike_rate: float)`
