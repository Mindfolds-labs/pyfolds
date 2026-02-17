# Quickstart

## Objetivo
Executar um exemplo mínimo com PyFolds.

## Escopo
Instanciação de configuração e execução do forward.

## Definições/Termos
- **Forward:** passagem de dados pela rede neural.

## Conteúdo técnico
```python
import torch
from pyfolds import MPJRDConfig, MPJRDNeuron

config = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=8)
model = MPJRDNeuron(config)
out = model(torch.randn(32, 4, 8))
print(out["spikes"])
```

## Referências
- [Examples](examples.md)
