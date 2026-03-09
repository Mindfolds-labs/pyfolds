# WaveMixin

`WaveMixin` adiciona oscilação multi-frequência com neuromodulação genérica e suporte a consolidação em modo `sleep`.

## Recursos
- Modulação oscilatória da entrada (`_wave_modulate`).
- Sincronia fase-potencial (`_compute_sync`).
- Neuromodulação por modo (`online`, `batch`, `sleep`, `inference`).
- Consolidação em sono com normalização e pruning (`_sleep_consolidation`).

## Exemplo rápido
```python
from pyfolds.core import MPJRDConfig, MPJRDNeuron
from pyfolds.advanced import WaveMixin

class WaveNeuron(WaveMixin, MPJRDNeuron):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.wave_enabled:
            self._init_wave(cfg)
```
