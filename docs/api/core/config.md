# API Core — `MPJRDConfig`

Dataclass imutável com hiperparâmetros de topologia, plasticidade, homeostase, neuromodulação e execução.

Exemplo:

```python
from pyfolds import MPJRDConfig
cfg = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=32)
```

Campos-chave:

- Topologia: `n_dendrites`, `n_synapses_per_dendrite`
- Estado estrutural: `n_min`, `n_max`, `w_scale`
- Plasticidade: `i_eta`, `i_gamma`, `i_ltp_th`, `i_ltd_th`
- Homeostase: `theta_init`, `homeostasis_eta`, `target_spike_rate`
- Execução: `plastic`, `defer_updates`, `dt`, `device`
