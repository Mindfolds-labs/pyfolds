# Quickstart PyFolds

```python
import torch
from pyfolds import MPJRDConfig, MPJRDNeuron

cfg = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=8)
neuron = MPJRDNeuron(cfg)

x = torch.randn(16, 4, 8)
out = neuron(x)
print(out['spike_rate'])
```
