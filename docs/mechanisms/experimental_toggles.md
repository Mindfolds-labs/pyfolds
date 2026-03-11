# Experimental toggles (safe-by-default)

## Objetivo
Permitir A/B controlado sem contaminar o baseline.

## Toggles
- `enable_phase_gating` (default: `False`): aplica gate de fase no delta STDP.
- `enable_dynamic_channel_gating` (default: `False`): ponto estrutural para gating por canal no STDP.
- `enable_wave_modulation` (default: `False`): reservado para modulação wave no forward.
- `enable_sleep_consolidation` (default: `True`, legado): consolidação durante sono já existente.
- `enable_dendritic_threshold_modulation` (default: `False`): reservado para modulação de limiar dendrítico.
- `debug_compare_baseline` (default: `False`): habilita execução comparativa A/B via helper.
- `debug_collect_mechanism_metrics` (default: `False`): inclui métricas do mecanismo em `forward`.

## Contrato
Use `MechanismToggleSet`/`ExperimentalMechanismConfig` em `pyfolds.advanced.experimental`.

## Modo comparativo
Use `compare_mechanism_vs_baseline(...)` para:
1. clonar estado inicial
2. rodar baseline e experimento
3. gerar diffs de saída
4. coletar métricas objetivas
