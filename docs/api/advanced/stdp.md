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
  - `classic`: LTD usa `pre_spikes * trace_post` (depressão condicionada ao spike pré)
  - `current`: LTD usa `post_spike * trace_post` (regra legada dependente de spike pós)

Diferença numérica entre modos (`ltd_rule`):
- Com `trace_post=1.0`, `A_minus=0.012`, limiar já superado e sem LTP:
  - `classic`, sem spike pré (`pre_spike=0`): `ΔLTD = 0.0`
  - `current`, com spike pós (`post_spike=1`): `ΔLTD = -0.012`
- Em entradas sub-limiar (sem pré-spike), `current` tende a produzir maior depressão acumulada do que `classic` quando há atividade pós-sináptica.
