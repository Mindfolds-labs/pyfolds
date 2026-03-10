# ADR-048: StatisticsAccumulator sparse-masked adaptativo e cache de pesos

## Status
Accepted

## Contexto
O caminho denso do `StatisticsAccumulator` é a referência científica porque preserva contagem uniforme por sinapse e minimiza viés de amostragem. Entretanto, em regimes altamente esparsos, acumular tudo densamente pode desperdiçar tempo de execução.

Também havia custo recorrente de `torch.stack([d.W ...])` no forward vetorizado do neurônio.

## Decisão
1. **Baseline denso preservado** como modo padrão (`stats_accumulator_mode="dense"`).
2. **Novo modo `sparse_masked`** usando máscara booleana sobre tensores densos (`abs(x) > activity_threshold`) antes de considerar `torch.sparse`, para reduzir risco arquitetural.
3. **Contagem por sinapse** (`synapse_sample_count`) para médias `x_mean` por denominador local e proteção contra divisão por zero (`clamp_min(1.0)`).
4. **Fallback adaptativo para denso** quando `activity_ratio > sparse_min_activity_ratio`, evitando regressão de performance em baixa esparsidade.
5. **Telemetria/profiling opt-in** para custo de acumulação: `accumulator_time_ms`, `activity_ratio`, `sparse_path_used`, `dense_fallback_used`, `nonzero_sample_ratio`.
6. **Cache de pesos consolidados no neurônio** com invalidade explícita (`invalidate_weight_cache`) para eliminar `torch.stack` redundante no forward.

## Consequências
- Melhor rastreabilidade científica (contagem local por sinapse + telemetria).
- Risco menor de viés estatístico em cenários com máscaras esparsas.
- Ganhos de performance mensuráveis em cenários de alta esparsidade e redução de overhead no forward via cache.
- Necessidade de invalidar cache após mutações de pesos (plasticidade e sono já cobertos).

## Métricas para validar ganho
- `accumulator_time_ms` médio por passo.
- Taxa de uso real do caminho sparse (`sparse_path_used`) e fallback denso.
- Throughput (samples/s) em diferentes `batch`, `shape`, `activity_threshold` e `active_ratio`.
- Desvio numérico vs baseline denso (`allclose` com tolerâncias explícitas).
