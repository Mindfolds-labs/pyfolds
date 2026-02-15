# API Core — `MPJRDSynapse`

Unidade sináptica com estado explícito:

- `N` (int): filamentos discretos (longo prazo)
- `I` (float): potencial interno (curto prazo)
- `W` (property): peso derivado de `N`

Métodos principais:

- `update(pre_rate, post_rate, R, dt, mode=None)`
- `consolidate(dt=1.0)`
