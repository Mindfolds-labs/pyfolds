# Phase Activity Observability

## Objetivo
Monitorar distribuição de atividade por fase para avaliar efeitos de circadiano/replay/pruning.

## Variáveis
- **Entrada:** atividade corrente, fase ativa e baseline acumulado.
- **Controle:** `circadian_enabled`, `circadian_phase_bins`, `circadian_auto_mode`.
- **Saída:** `active_phase_idx`, histograma de fase, `delta_vs_baseline`.

## Fluxo
1. Registrar atividade no bin de fase atual.
2. Atualizar baseline de referência.
3. Emitir relatório de desvio para monitoramento.

## Custo computacional
O(B) para atualização/leitura de bins de fase; memória pequena e fixa por configuração.

## Integração
- `MPJRDNeuron.collect_phase_activity_report` (`src/pyfolds/core/neuron.py`).
- `MPJRDNeuron.get_metrics` inclui contexto de fase (`src/pyfolds/core/neuron.py`).
- Parâmetros circadianos em `MPJRDConfig` (`src/pyfolds/core/config.py`).

## Estado
- **Rótulo:** `Estável`.
- **Justificativa:** relatório é observável, de baixo risco e já integrado ao ciclo normal.
