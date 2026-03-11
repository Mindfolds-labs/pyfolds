# Data Buffer State Strategy

## Objetivo
Definir estratégia de buffers e snapshots para manter rastreabilidade de estado sem bloquear o forward.

## Variáveis
- **Entrada:** eventos de execução, métricas de estado e snapshots periódicos.
- **Controle:** janelas de coleta e frequência de inspeção.
- **Saída:** buffers de observabilidade e dicionários de diagnóstico.

## Fluxo
1. Coletar métricas ao longo da execução.
2. Compactar estado em snapshots leves.
3. Disponibilizar relatórios para comparação baseline/experimento.

## Custo computacional
Sobrecarga linear no número de métricas armazenadas; custo controlado por frequência de coleta.

## Integração
- `StatisticsAccumulator` e `create_accumulator_from_config` (`src/pyfolds/core/accumulator.py`).
- `MPJRDNeuron.get_metrics` e `MPJRDNeuron.get_audit_trace_snapshot` (`src/pyfolds/core/neuron.py`).
- `compare_mechanism_vs_baseline`/`diff_output_stats` para diffs de saída (`src/pyfolds/advanced/experimental.py`).

## Estado
- **Rótulo:** `Estável`.
- **Justificativa:** mecanismo usa APIs de telemetria já consolidadas no pipeline core.
