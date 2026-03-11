# Replay e consolidação de poda

## Objetivo do mecanismo
Permitir consolidação de pruning após replay offline.

## Base científica resumida
Replay durante repouso/sono é frequentemente associado, de forma aproximada, a reforço e seleção de traços mnêmicos.

> **Nota de escopo científico**: o comportamento descrito aqui é uma aproximação computacional experimental de replay/consolidação e não deve ser interpretado como equivalência biológica completa.

## Tradução computacional adotada
`run_replay_cycle()` aplica replay comprimido por elegibilidade e, quando habilitado, chama `_consolidate_pruning_from_runtime()` para atualizar `pruning_mask` a partir de pesos consolidados e limiar de runtime.

A consolidação usa o snapshot de conectividade/poda no neurônio, permitindo análise de efeito pré/pós replay via snapshots.

## Arquivos do código afetados
- `src/pyfolds/core/neuron.py`
- `src/pyfolds/core/config.py`

## Flags de ativação/desativação
Flags experimentais relevantes:
- `consolidate_pruning_after_replay` (default: `False`)
- `pruning_enabled`
- `pruning_strategy` (`static`, `phase_scheduled`, `replay_consolidated`)
- `pruning_runtime_threshold`
- `replay_interval_steps`

## Riscos de implementação
- Consolidação prematura sob baixo sinal.
- Sensibilidade a limiar (`pruning_runtime_threshold`) em diferentes regimes.
- Drift de conectividade caso replay seja muito frequente sem validação externa.

## Estratégia de teste
Cobertura por snapshots e execução de forward/replay:
- comparar `collect_pruning_snapshot()` antes/depois;
- validar variação de `pruned_ratio` apenas quando flags experimentais estiverem ativas.

## Critérios de observabilidade/debug
- `collect_pruning_snapshot()` (`pruning_mask`, `pruned_by_dendrite`, `pruned_ratio`).
- `collect_connectivity_snapshot()` (`effective_connectivity`, `active_ratio`).
- `get_metrics()` para telemetria contextual (`effective_replay_priority`, `effective_consolidation_rate`, gates circadianos).

## Comportamento offline (consolidação/replay)
O replay ocorre em ciclo offline/sono e pode elevar elegibilidade de traços. Quando `consolidate_pruning_after_replay=True`, a consolidação de poda é executada no mesmo ciclo, atualizando a máscara efetiva sem depender de interação online imediata.

## Baseline safety
Com flags experimentais desligadas (especialmente `consolidate_pruning_after_replay=False`), o caminho estável é preservado: replay não força consolidação de poda pós-ciclo, mantendo o comportamento padrão de pruning configurado.
