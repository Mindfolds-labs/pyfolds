# Cooperative Integration (NeuronV2)

Este documento deixa um **prompt pronto** para evoluir o modelo dendrítico sem substituir o neurônio atual.

## Prompt pronto para Codex

```text
Objetivo: criar uma versão experimental MPJRDNeuronV2 com integração dendrítica cooperativa (Soft-WTA), mantendo MPJRDNeuron intacto para comparação A/B.

Tarefas:
1) Criar `src/pyfolds/core/neuron_v2.py` com a classe `MPJRDNeuronV2` (herdando de MPJRDNeuron).
2) Implementar `forward` vetorizado:
   - W = log2(1+N)/w_scale
   - v_dend = einsum('ds,bds->bd', W, x)
   - dendritic_gain = sigmoid(v_dend - theta*0.5)
   - somatic = dendritic_gain.sum(dim=-1)
   - spikes = (somatic >= theta).float()
3) Retornar no output: `somatic`, `dendritic_gain`, `v_dend`, `spikes`, e manter `u`/`gated` por compatibilidade (`u=somatic`, `gated=dendritic_gain`).
4) Em batch mode com `defer_updates`, acumular `dendritic_gain` no `StatisticsAccumulator`.
5) Exportar `MPJRDNeuronV2` em `pyfolds.core` e em `pyfolds` raiz, incluindo factory `create_neuron_v2`.
6) Adicionar testes unitários para shape de saída e para confirmar que múltiplos dendritos podem contribuir ao mesmo spike.

Critérios de aceite:
- Todos os testes relevantes passam.
- MPJRDNeuron original continua com comportamento WTA.
- Nova versão pode ser instanciada diretamente: `from pyfolds import MPJRDNeuronV2`.
```

## Estratégia de rollout

1. Testar com `MPJRDNeuronV2` em paralelo ao modelo atual.
2. Comparar métricas de spike rate, robustez a ruído e saturação.
3. Migrar gradualmente apenas se houver ganho consistente.
