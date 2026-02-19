# API Advanced — STDP

Plasticidade dependente do tempo relativo de spikes.

Parâmetros:
- `plasticity_mode`
- `tau_pre`, `tau_post`
- `A_plus`, `A_minus`
- `stdp_input_source`: `"raw" | "stp"`
  - `raw`: detecta pré-spike no sinal original antes de STP
  - `stp`: detecta pré-spike no sinal já modulado por STP (retrocompatível padrão)
- `ltd_rule`: `"classic" | "current"`
  - `classic`: LTD usa `pre_spikes * trace_post`
  - `current`: LTD usa regra legada dependente de spike pós
