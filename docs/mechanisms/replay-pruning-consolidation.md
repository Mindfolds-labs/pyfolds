# Replay e consolidação de poda

## Objetivo do mecanismo
Permitir consolidação de pruning após replay offline.

## Base científica resumida
Replay durante sono é associado a reforço e seleção de traços mnêmicos.

## Tradução computacional adotada
`run_replay_cycle()` pode chamar `_consolidate_pruning_from_runtime()` quando habilitado.

## Arquivos do código afetados
- `src/pyfolds/core/neuron.py`
- `src/pyfolds/core/config.py`

## Flags de ativação/desativação
`consolidate_pruning_after_replay`.

## Riscos de implementação
Consolidação prematura sob baixo sinal.

## Estratégia de teste
Cobertura indireta por snapshots e execução de forward/replay.

## Critérios de observabilidade/debug
Métricas de `pruned_ratio` antes/depois do replay.
