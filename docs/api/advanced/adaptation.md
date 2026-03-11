# API Advanced — Adaptation

Mecanismo de spike-frequency adaptation (SFA).

Parâmetros relevantes em `MPJRDConfig`:
- `adaptation_enabled`
- `adaptation_increment`
- `adaptation_decay`
- `adaptation_max`

Semântica:
- SFA é aplicada antes do threshold no core de decisão (`u_eff = u - I_adapt`), no ponto de decisão do fluxo avançado.
- No mixin avançado atual, `AdaptationMixin.forward` delega para `super().forward(...)` e a atualização de `u`/`I_adapt` associada ao spike confirmado ocorre **após** o `super()`, via integração com `RefractoryMixin`.
- A decisão final de spike permanece no estágio pós-refratário (`final_spikes`), e não é sobrescrita pelo mixin de adaptação.

Referências de implementação:
- `src/pyfolds/advanced/adaptation.py`
- `src/pyfolds/advanced/refractory.py`
- `src/pyfolds/advanced/stdp.py`
- `src/pyfolds/advanced/short_term.py`
- `src/pyfolds/advanced/__init__.py`
