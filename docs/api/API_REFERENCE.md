# API Reference — PyFolds

> Referência manual das APIs principais (core + wave), alinhada aos docstrings do código em `src/pyfolds`.

## 1. `MPJRDConfig`

Classe: `pyfolds.core.config.MPJRDConfig`

### 1.1 Parâmetros

#### Topologia
- `n_dendrites: int = 4`
- `n_synapses_per_dendrite: int = 32`

#### Filamentos (`N`)
- `n_min: int = 0`
- `n_max: int = 31`
- `w_scale: float = 5.0`

#### Plasticidade (`I`)
- `i_eta: float = 0.01`
- `i_gamma: float = 0.99`
- `beta_w: float = 0.0`
- `i_ltp_th: float = 5.0`
- `i_ltd_th: float = -5.0`
- `ltd_threshold_saturated: float = -10.0`
- `i_min: float = -20.0`
- `i_max: float = 50.0`
- `i_decay_sleep: float = 0.99`

#### Dinâmica de curto prazo
- `u0: float = 0.1`
- `R0: float = 1.0`
- `U: float = 0.2`
- `tau_fac: float = 100.0`
- `tau_rec: float = 800.0`
- `saturation_recovery_time: float = 60.0`

#### Homeostase
- `theta_init, theta_min, theta_max`
- `homeostasis_alpha, homeostasis_eta`
- `target_spike_rate`
- `dead_neuron_threshold`, `dead_neuron_penalty`
- `activity_threshold`, `homeostasis_eps`

#### Mecanismos avançados
- Backprop: `backprop_enabled`, `backprop_delay`, `backprop_signal`, `backprop_amp_tau`, `backprop_trace_tau`, `backprop_max_amp`, `backprop_max_gain`
- Adaptação: `adaptation_enabled`, `adaptation_increment`, `adaptation_decay`, `adaptation_max`, `adaptation_tau`
- Refratário: `refrac_mode`, `t_refrac_abs`, `t_refrac_rel`, `refrac_rel_strength`
- Inibição: `inhibition_mode`, `lateral_strength`, `feedback_strength`, `inhibition_sigma`, `n_excitatory`, `n_inhibitory`, `target_sparsity`
- STDP: `plasticity_mode`, `tau_pre`, `tau_post`, `A_plus`, `A_minus`

#### Neuromodulação
- `neuromod_mode`: `external | capacity | surprise`
- `neuromod_scale`
- Capacidade: `cap_k_sat`, `cap_k_rate`, `cap_bias`
- Surpresa: `sup_k`, `sup_bias`

#### Execução
- `plastic`, `defer_updates`, `consolidation_rate`, `eps`, `dt`, `device`

### 1.2 Métodos utilitários
- `get_ts(param_name) -> float`
- `get_decay_rate(tau, dt=None) -> float`
- `to_dict() -> Dict`
- `from_dict(data) -> MPJRDConfig`
- `get_preset(name='default') -> MPJRDConfig`

### Exemplo

```python
from pyfolds import MPJRDConfig

cfg = MPJRDConfig(
    n_dendrites=4,
    n_synapses_per_dendrite=16,
    neuromod_mode="capacity",
    defer_updates=True,
)
print(cfg.to_dict()["n_max"])  # 31
```

---

## 2. `MPJRDNeuron`

Classe: `pyfolds.core.neuron.MPJRDNeuron`

### 2.1 `forward`

```python
forward(
    x,
    reward=None,
    mode=None,
    collect_stats=True,
    dt=1.0,
) -> Dict[str, torch.Tensor]
```

**Entrada esperada**
- `x`: tensor `[batch, n_dendrites, n_synapses_per_dendrite]`.

**Saída (campos principais)**
- `spikes`: disparos `[batch]`
- `u`: potencial somático `[batch]`
- `v_dend`: potencial por dendrito `[batch, D]`
- `gated`: ativação após competição/gating `[batch, D]`
- `theta`, `r_hat`
- `spike_rate`, `saturation_ratio`
- `R` (sinal neuromodulador)
- métricas agregadas (`N_mean`, `I_mean`, `W_mean`)

### 2.2 `apply_plasticity`

```python
apply_plasticity(dt=1.0, reward=None) -> None
```

Aplica atualização sináptica (tipicamente após acumulação em modo batch/defer), combinando atividade pré/pós e sinal `R`.

### 2.3 `sleep`

```python
sleep(duration=60.0) -> None
```

Executa consolidação offline, favorecendo estabilização de memória estrutural.

### 2.4 `get_metrics`

```python
get_metrics() -> Dict[str, Any]
```

Retorna snapshot de métricas do neurônio e do estado homeostático/sináptico.

### Exemplo

```python
import torch
from pyfolds import MPJRDConfig, MPJRDNeuron

cfg = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=8)
neuron = MPJRDNeuron(cfg)

x = torch.rand(32, 4, 8)
out = neuron(x, reward=0.2)
print(out["spikes"].shape)

neuron.apply_plasticity(dt=1.0, reward=0.2)
neuron.sleep(duration=30.0)
print(neuron.get_metrics().keys())
```

---

## 3. `MPJRDWaveNeuron`

Classe: `pyfolds.wave.neuron.MPJRDWaveNeuron`

### 3.1 Configuração wave (`MPJRDWaveConfig`)

Parâmetros adicionais:
- `wave_enabled`
- `base_frequency`, `frequency_step`, `class_frequencies`
- `phase_decay`, `phase_buffer_size`
- `phase_sensitivity`, `phase_plasticity_gain`
- `dendritic_threshold`
- `latency_scale`, `amplitude_eps`

### 3.2 `forward` (wave)

```python
forward(
    x,
    reward=None,
    mode=None,
    collect_stats=True,
    target_class=None,
) -> Dict[str, torch.Tensor]
```

Campos adicionais no retorno:
- `phase`, `latency`, `amplitude`, `frequency`
- `phase_sync`, `arrival_leading`
- `wave_real`, `wave_imag`, `wave_complex`

### 3.3 `apply_plasticity` (wave)

```python
apply_plasticity(dt=1.0, reward=None) -> None
```

No v3.0, `reward` pode ser modulado por sincronização de fase (`last_phase_sync`) antes de delegar à regra base.

### Exemplo

```python
import torch
from pyfolds import MPJRDWaveConfig, MPJRDWaveNeuron

cfg = MPJRDWaveConfig(
    n_dendrites=4,
    n_synapses_per_dendrite=8,
    base_frequency=12.0,
    frequency_step=4.0,
)

neuron = MPJRDWaveNeuron(cfg)
x = torch.rand(16, 4, 8)
out = neuron(x, target_class=2, reward=0.1)

print(out["phase"].shape)
print(out["wave_complex"].dtype)
```

---

## 4. Observações de uso

- Prefira `mode=INFERENCE` quando não quiser alterar homeostase/plasticidade.
- Para workloads grandes, combine `defer_updates=True` + `apply_plasticity()` em janelas.
- Em cenários de classificação temporal, use `MPJRDWaveNeuron`/`MPJRDWaveLayer` para explorar codificação por fase/frequência.
