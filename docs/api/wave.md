# API Reference - Módulo Wave

Documentação do módulo de codificação por fase e amplitude (v3.0).

## Classes principais
- `MPJRDWaveNeuron`
- `MPJRDWaveLayer`
- `MPJRDWaveNetwork`

## Conceitos
- Frequência base aprendível por neurônio.
- Fase como representação temporal de correlação.
- Amplitude logarítmica como medida de confiança.

## Exemplo

```python
from pyfolds.wave import MPJRDWaveNeuron

neuron = MPJRDWaveNeuron()
out = neuron(x)
phase = out['phase']
amplitude = out['amplitude']
```
