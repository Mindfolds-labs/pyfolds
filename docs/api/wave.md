# API Reference - Módulo Wave

Documentação do módulo de codificação por fase e amplitude (v3.0).

## Preferência atual

> **Caminho recomendado:** usar `WaveMixin + MPJRDNeuron` (stack `pyfolds.advanced`) para novos projetos.
>
> A implementação `MPJRDWaveNeuron` permanece disponível apenas por compatibilidade e está **depreciada**.

## Classes principais
- `MPJRDWaveConfig` *(legado/deprecated)*
- `MPJRDWaveNeuron` *(legado/deprecated)*
- `MPJRDWaveLayer`
- `MPJRDWaveNetwork`

## Conceitos
- Frequência base aprendível por neurônio.
- Fase como representação temporal de correlação.
- Amplitude logarítmica como medida de confiança.

## Exemplo (compatibilidade legada)

```python
import torch
from pyfolds.wave import MPJRDWaveConfig, MPJRDWaveNeuron

cfg = MPJRDWaveConfig(
    n_dendrites=4,
    n_synapses_per_dendrite=8,
    base_frequency=12.0,
    frequency_step=4.0,
)

neuron = MPJRDWaveNeuron(cfg)
x = torch.randn(2, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
out = neuron(x)

phase = out["phase"]
amplitude = out["amplitude"]
```

## Compatibilidade (`MPJRDWaveNeuron`)

- `MPJRDWaveNeuron` requer `cfg: MPJRDWaveConfig` no construtor.
- Tanto `MPJRDWaveNeuron` quanto `MPJRDWaveConfig` emitem `DeprecationWarning`.
- Para evolução de código, migre para `MPJRDNeuron` com capacidades wave via `WaveMixin`.
