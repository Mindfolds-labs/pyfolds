# Quickstart

```python
import torch
from pyfolds import MPJRDConfig, MPJRDNeuron

cfg = MPJRDConfig(
    n_dendrites=4,
    n_synapses_per_dendrite=8,
    plastic=True,
)

neuron = MPJRDNeuron(cfg)
x = torch.randn(32, 4, 8)
out = neuron(x, reward=0.2)

print(out["spikes"].shape)
print(out["v_dend"].shape)
print(float(out["spike_rate"]))
```

## Modos de execução

- `LearningMode.ONLINE`: atualiza online.
- `LearningMode.BATCH`: acumula e aplica plasticidade em lote.
- `LearningMode.SLEEP`: consolidação offline.
- `LearningMode.INFERENCE`: inferência sem aprendizado.
