# Observabilidade de atividade por fase

## Objetivo do mecanismo
Monitorar distribuição de atividade por fase e comparar baseline vs ativo.

## Base científica resumida
Fase oscilatória organiza janelas de excitabilidade/plasticidade.

## Tradução computacional adotada
Histograma de atividade por `circadian_phase_bins` em buffer não persistente.

## Arquivos do código afetados
- `src/pyfolds/core/neuron.py`

## Flags de ativação/desativação
Herdado de configuração circadiana e estratégia de pruning por fase.

## Riscos de implementação
Perda de detalhe temporal por discretização.

## Estratégia de teste
Executar forwards e validar presença de `delta_vs_baseline` no relatório.

## Critérios de observabilidade/debug
`collect_phase_activity_report()`.
