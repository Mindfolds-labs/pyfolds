# Phase gating experiment (STDP)

## Regra implementada
Com `enable_phase_gating=True`:

`delta_w = base_delta_w * max(0, cos(phase))`

Com toggle desligado, o update STDP permanece igual ao baseline.

## Ponto de integração
`STDPMixin._update_stdp_traces`.

## Métricas coletadas
- `spike_rate`
- `average_weight_update`
- `dendritic_activity_rate`
- `refractory_block_rate`
- `adaptation_level_mean`
- `phase_alignment_mean`
- `sparsity_ratio`
- `active_dendrite_ratio`
- `learning_event_count`

## Riscos/limites
- `dynamic_channel_gating` está em modo mínimo e não deve ser usado como mecanismo validado ainda.
- `enable_wave_modulation` e `enable_dendritic_threshold_modulation` estão apenas com ponto estrutural nesta etapa.
