# Revisão crítica de aderência científica

## Objetivo do mecanismo
Avaliar o que é alinhado à literatura versus aproximação de engenharia.

## Base científica resumida
Referências conceituais: oscillatory coordination, theta-gamma coding, cortical tracking, STDP, replay, pruning.

## Tradução computacional adotada
Implementações focam controle de estado, máscaras vetorizadas e relatórios operacionais.

## Arquivos do código afetados
- `src/pyfolds/core/neuron.py`
- `src/pyfolds/core/config.py`
- `src/pyfolds/advanced/noetic_model.py`

## Flags de ativação/desativação
`pruning_enabled`, `pruning_strategy`, `consolidate_pruning_after_replay`.

## Riscos de implementação
Confundir analogia computacional com equivalência biológica.

## Estratégia de teste
Combinar testes unitários com benchmarks e ablações controladas.

## Critérios de observabilidade/debug
Comparar baseline vs ativo por snapshots de conectividade, fase e engrams.
