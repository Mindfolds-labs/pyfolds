# Debug and Observability Playbook

## Objetivo
Padronizar inspeção de mecanismos para facilitar diagnóstico de regressão e validação científica.

## Variáveis
- **Entrada:** saídas de forward, snapshots de fase/conectividade e traces de auditoria.
- **Controle:** toggles de mecanismo e escopo do experimento.
- **Saída:** relatórios comparativos e evidências de comportamento.

## Fluxo
1. Executar baseline e variante com mesmos estímulos.
2. Coletar snapshots por mecanismo (fase, pruning, engram).
3. Comparar diffs e registrar interpretação do risco.

## Custo computacional
Custo extra proporcional ao número de execuções comparadas (tipicamente 2x para baseline+experimento).

## Integração
- `collect_mechanism_report` e `compare_mechanism_vs_baseline` (`src/pyfolds/advanced/experimental.py`).
- `MPJRDNeuron.collect_phase_activity_report` (`src/pyfolds/core/neuron.py`).
- `MPJRDNeuron.collect_connectivity_snapshot` e `MPJRDNeuron.collect_engram_report` (`src/pyfolds/core/neuron.py`).

## Estado
- **Rótulo:** `Estável`.
- **Justificativa:** o playbook usa pontos de observabilidade já expostos por APIs públicas internas.
