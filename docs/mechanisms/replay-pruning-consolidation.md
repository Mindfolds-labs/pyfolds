# Replay Pruning Consolidation

## Objetivo
Consolidar poda após replay offline para refletir traços reforçados na conectividade efetiva.

## Variáveis
- **Entrada:** elegibilidade de replay, pesos correntes e estado de pruning.
- **Controle:** `consolidate_pruning_after_replay`, `replay_interval_steps`, `pruning_runtime_threshold`.
- **Saída:** `pruning_mask` consolidada e métricas de conectividade pós-replay.

## Fluxo
1. Executar ciclo de replay no neurônio.
2. Opcionalmente consolidar pruning a partir do estado runtime.
3. Coletar snapshots pré/pós para auditoria de efeito.

## Custo computacional
Custo adicional depende da frequência de replay; atualização de máscara é O(S) nas sinapses relevantes.

## Integração
- `MPJRDNeuron.run_replay_cycle` (`src/pyfolds/core/neuron.py`).
- `MPJRDNeuron._consolidate_pruning_from_runtime` (`src/pyfolds/core/neuron.py`).
- `MPJRDNeuron.collect_pruning_snapshot` e `collect_connectivity_snapshot` (`src/pyfolds/core/neuron.py`).
- Flags em `MPJRDConfig` (`src/pyfolds/core/config.py`).

## Estado
- **Rótulo:** `Experimental`.
- **Justificativa:** comportamento depende de combinação de flags e sensibilidade de limiar em regime offline.
