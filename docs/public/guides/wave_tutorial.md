# Guia Introdutório — MPJRD-Wave v3.0

A versão v3.0 adiciona codificação por **fase** e **frequência** ao pipeline MPJRD.

## 1. Conceitos-chave

- **Amplitude (`amplitude`)**: intensidade efetiva da integração somática.
- **Fase (`phase`)**: posição angular do disparo dentro do ciclo oscilatório.
- **Frequência (`frequency`)**: portadora dinâmica (global ou por classe).
- **Latência (`latency`)**: proxy temporal associada à amplitude.

## 2. Configuração mínima

```python
from pyfolds import MPJRDWaveConfig

cfg = MPJRDWaveConfig(
    n_dendrites=4,
    n_synapses_per_dendrite=16,
    base_frequency=12.0,
    frequency_step=4.0,
    phase_sensitivity=1.0,
    phase_plasticity_gain=0.25,
)
```

## 3. Uso com `MPJRDWaveNeuron`

```python
import torch
from pyfolds import MPJRDWaveNeuron

neuron = MPJRDWaveNeuron(cfg)
x = torch.rand(32, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
out = neuron(x, reward=0.2, target_class=3)

print(out.keys())
print("spike_rate:", out["spike_rate"].item())
print("phase mean:", out["phase"].mean().item())
print("frequency:", out["frequency"].item())
```

## 4. Camada e rede wave

```python
from pyfolds import MPJRDWaveLayer, MPJRDWaveNetwork

layer = MPJRDWaveLayer(n_neurons=16, cfg=cfg, name="wave_hidden")
net = MPJRDWaveNetwork().add_wave_layer("wave_hidden", n_neurons=16, cfg=cfg)
```

## 5. Estratégias de frequência

### 5.1 Frequência linear por classe

```python
cfg = MPJRDWaveConfig(base_frequency=10.0, frequency_step=3.0)
# classe k -> f = 10 + 3*k
```

### 5.2 Frequências explícitas por classe

```python
cfg = MPJRDWaveConfig(class_frequencies=(8.0, 10.0, 12.0, 14.0, 16.0))
```

## 6. Plasticidade sensível à fase

No `apply_plasticity`, a recompensa pode ser modulada por sincronização de fase (`last_phase_sync`):

- sincronização alta → reforço maior;
- dessincronização → reforço menor.

Isso adiciona um canal temporal ao aprendizado além de taxa de spike.

## 7. Diagnóstico recomendado

Monitore por batch/época:

- `spike_rate`
- `phase.mean()` e `phase.std()`
- `frequency`
- `phase_sync`
- saturação (`N == n_max`)

Se `phase_sync` oscilar demais, reduza `phase_plasticity_gain` ou aumente `phase_decay`.
