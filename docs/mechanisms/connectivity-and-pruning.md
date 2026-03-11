# Conectividade estrutural e poda

## Objetivo do mecanismo
Representar fluxo estrutural e poda com máscaras explícitas e vetorizadas.

## Base científica resumida
Refinamento sináptico é central em aprendizagem e estabilização de circuitos.

## Tradução computacional adotada
Máscara efetiva = `connectivity_mask * pruning_mask` aplicada aos pesos consolidados.

## Arquivos do código afetados
- `src/pyfolds/core/neuron.py`

## Flags de ativação/desativação
`pruning_strategy`: `static`, `phase_scheduled`, `replay_consolidated`.

## Riscos de implementação
Aproximação por magnitude não captura toda causalidade de plasticidade.

## Estratégia de teste
Teste com mascaramento parcial e comparação de potenciais dendríticos.

## Critérios de observabilidade/debug
`collect_connectivity_snapshot()` e `collect_pruning_snapshot()`.
