# Connectivity and Pruning

## Objetivo
Controlar conectividade efetiva via poda para reduzir custo e manter sinapses relevantes.

## Variáveis
- **Entrada:** pesos sinápticos, atividade e estado de replay.
- **Controle:** `pruning_enabled`, `pruning_strategy`, `pruning_runtime_threshold`.
- **Saída:** `pruning_mask`, `active_ratio`, `effective_connectivity`.

## Fluxo
1. Atualizar máscara de poda com regra ativa.
2. Aplicar máscara à conectividade durante o forward.
3. Expor snapshots para auditoria e comparação com baseline.

## Custo computacional
O(S) sobre número de sinapses para atualizar/aplicar máscara; custo extra de memória para snapshots de diagnóstico.

## Integração
- `MPJRDNeuron._refresh_pruning_mask` e `MPJRDNeuron._phase_pruning_gate` (`src/pyfolds/core/neuron.py`).
- `MPJRDNeuron.collect_connectivity_snapshot` e `MPJRDNeuron.collect_pruning_snapshot` (`src/pyfolds/core/neuron.py`).
- Campos de poda em `MPJRDConfig` (`src/pyfolds/core/config.py`).

## Estado
- **Rótulo:** `Estável`.
- **Justificativa:** fluxo principal de máscara e snapshots já está integrado ao caminho padrão do neurônio.
