# Guia de Imports e API ativa (`pyfolds`)

Este guia resume **como importar** o pacote e quais classes/funções estão ativas na API pública (`pyfolds.__all__`).

## Import recomendado

```python
import pyfolds

cfg = pyfolds.MPJRDConfig(n_dendrites=4)
neuron = pyfolds.MPJRDNeuron(cfg)
```

Também é suportado o import seletivo:

```python
from pyfolds import MPJRDConfig, MPJRDNeuron, MPJRDNetwork
```

## Componentes públicos principais

### Core
- `MPJRDConfig`
- `MPJRDNeuron`
- `MPJRDNeuronV2`
- `MPJRDLayer`
- `MPJRDNetwork`
- `MPJRDWaveLayer`
- `MPJRDWaveNetwork`
- `NetworkBuilder`
- `MPJRDWaveConfig`
- `MPJRDWaveNeuron`
- `NeuronFactory`
- `NeuronType`

### Serialização
- `VersionedCheckpoint`
- `FoldReader`
- `FoldWriter`
- `FoldSecurityError`
- `save_fold_or_mind`
- `load_fold_or_mind`
- `peek_fold_or_mind`
- `peek_mind`
- `read_nuclear_arrays`
- `is_mind`
- `NoECC`
- `ReedSolomonECC`
- `ecc_from_protection`

### Monitoring e tipos
- `HealthStatus`
- `NeuronHealthCheck`
- `LearningMode`
- `ConnectionType`
- `learning_mode`

### Telemetry
- `TelemetryController`
- `TelemetryConfig`
- `Sink`, `NoOpSink`, `MemorySink`, `ConsoleSink`, `JSONLinesSink`, `DistributorSink`
- `forward_event`, `commit_event`, `sleep_event`
- `forward_event_lazy`, `commit_event_lazy`, `sleep_event_lazy`
- `RingBuffer`
- `telemetry`
- `ForwardPayload`, `CommitPayload`, `SleepPayload`

## Observação sobre componentes avançados

Quando as dependências do módulo avançado estão disponíveis, estes símbolos também ficam públicos:
- `MPJRDNeuronAdvanced`
- `MPJRDLayerAdvanced`
- `MPJRDWaveNeuronAdvanced`
- `MPJRDWaveLayerAdvanced`

## Contrato de estabilidade para aplicações consumidoras

Para aplicações que dependem de `from pyfolds import ...`, use sempre símbolos em `pyfolds.__all__`.
As verificações automatizadas em `tests/unit/test_public_import_surface.py` exercitam esse contrato para evitar regressões de import.
