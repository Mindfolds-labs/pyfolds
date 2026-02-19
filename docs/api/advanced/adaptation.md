# API Advanced — Adaptation

Mecanismo de spike-frequency adaptation (SFA).

Parâmetros relevantes em `MPJRDConfig`:
- `adaptation_enabled`
- `adaptation_increment`
- `adaptation_decay`
- `adaptation_max`

Semântica:
- SFA é aplicada antes do threshold no core (`u_eff = u - I_adapt`).
- `AdaptationMixin` atualiza apenas estado adaptativo e não sobrescreve spike final pós-refratário.
