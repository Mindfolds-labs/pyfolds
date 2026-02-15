# API Core — `StatisticsAccumulator`

Acumula estatísticas em batch para atualização posterior.

## Métodos principais

- `accumulate(x, gated, spikes, ...)`
- `get_averages()`
- `reset()`
- `enable_history(enabled=True)`

Uso comum: `LearningMode.BATCH` com `defer_updates=True`.
