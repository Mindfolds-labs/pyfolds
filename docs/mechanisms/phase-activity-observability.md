# Observabilidade de atividade por fase

## Objetivo do mecanismo
Monitorar distribuição de atividade por fase e comparar baseline vs ativo.

## Base científica resumida
Fase oscilatória pode organizar janelas de excitabilidade/plasticidade.

> **Nota de escopo científico**: esta implementação é uma aproximação computacional experimental para observabilidade de dinâmica temporal; não representa equivalência biológica completa.

## Tradução computacional adotada
`collect_phase_activity_report()` resume o histograma acumulado por `circadian_phase_bins` em buffer não persistente e expõe:
- `phase_bins`
- `activity`
- `baseline`
- `active_phase_idx`
- `delta_vs_baseline`

Também é possível correlacionar o relatório com telemetria operacional (`get_metrics()`), incluindo `circadian_phase`, gates circadianos e métricas efetivas de política (`effective_replay_priority`, `effective_consolidation_rate`).

## Arquivos do código afetados
- `src/pyfolds/core/neuron.py`
- `src/pyfolds/core/config.py`

## Flags de ativação/desativação
Principais flags experimentais de contexto:
- `circadian_enabled`
- `circadian_phase_bins`
- `circadian_auto_mode`
- `replay_interval_steps`
- `pruning_strategy` (`static`, `phase_scheduled`, `replay_consolidated`)

## Riscos de implementação
- Perda de detalhe temporal fino por discretização em bins.
- Interpretação causal indevida entre fase e performance sem controle experimental.

## Estratégia de teste
Executar forwards em múltiplas etapas e validar presença e consistência de `delta_vs_baseline` e `active_phase_idx`.

## Critérios de observabilidade/debug
- `collect_phase_activity_report()` para inspeção local.
- `get_metrics()` para contexto de telemetria associado ao estado circadiano.

## Comportamento offline (consolidação/replay)
Em cenários com replay periódico (incluindo janelas de sono), o histograma de fase continua acumulando atividade observada em runtime; efeitos indiretos de replay/consolidação podem aparecer como deslocamentos de `active_phase_idx` e de `delta_vs_baseline`, sem persistência automática do histograma entre sessões.

## Baseline safety
Com flags experimentais desligadas (por exemplo `circadian_enabled=False` e estratégia de pruning padrão), o caminho estável permanece preservado: execução regular com relatório de fase disponível sem acionar comportamento adicional de replay/consolidação orientado por fase.
