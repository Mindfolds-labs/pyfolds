# Estratégia de dados, buffers e estado estrutural

## Objetivo do mecanismo
Separar parâmetros, buffers persistentes, buffers de runtime e dados externos.

## Base científica resumida
Modelos neurais com estado interno estável e estado dinâmico separado facilitam reprodutibilidade.

## Tradução computacional adotada
`connectivity_mask`/`pruning_mask` como buffers persistentes; traces e caches de runtime com `persistent=False`.

## Arquivos do código afetados
- `src/pyfolds/core/neuron.py`
- `src/pyfolds/core/config.py`

## Flags de ativação/desativação
`pruning_enabled`, `pruning_strategy`, `consolidate_pruning_after_replay`.

## Riscos de implementação
Threshold inadequado pode podar demais.

## Estratégia de teste
Testes unitários validando presença/ausência em `state_dict` e efeito da máscara no forward.

## Critérios de observabilidade/debug
Snapshots de conectividade/poda/atividade por fase.
