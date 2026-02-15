# API Core — `MPJRDNeuron`

Classe principal do PyFolds.

## Assinatura

```python
MPJRDNeuron(cfg: MPJRDConfig, enable_telemetry: bool = False, telemetry_profile: str = "off", name: str | None = None)
```

## Métodos principais

- `forward(x, reward=None, mode=None, collect_stats=True) -> dict`
- `set_mode(mode: LearningMode) -> None`
- `apply_plasticity() -> dict`
- `sleep(duration=100.0, dt=1.0) -> dict`
- `reset_state() -> None`

## Saída típica de `forward`

- `spikes`, `u`, `v_dend`, `gated`
- `theta`, `r_hat`, `R`
- `spike_rate`, `saturation_ratio`
- `N_mean`, `W_mean`, `I_mean`
