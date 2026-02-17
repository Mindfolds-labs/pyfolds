# Mapa de Código do pyfolds (Sheer baseline)

- Módulos analisados: **51**
- Dependências internas (imports `pyfolds.*`): **65**

## Pacotes
- `pyfolds`: 11 módulo(s)
- `pyfolds.advanced`: 7 módulo(s)
- `pyfolds.core`: 10 módulo(s)
- `pyfolds.layers`: 2 módulo(s)
- `pyfolds.monitoring`: 1 módulo(s)
- `pyfolds.network`: 3 módulo(s)
- `pyfolds.serialization`: 3 módulo(s)
- `pyfolds.telemetry`: 6 módulo(s)
- `pyfolds.utils`: 6 módulo(s)
- `pyfolds.wave`: 2 módulo(s)

## Módulos e símbolos
### `pyfolds`
- Arquivo: `__init__.py`
- Classes: (nenhuma)
- Funções de topo: (nenhuma)
- Imports internos: (nenhum)

### `pyfolds.advanced`
- Arquivo: `advanced/__init__.py`
- Classes: MPJRDNeuronAdvanced, MPJRDWaveNeuronAdvanced, MPJRDLayerAdvanced, MPJRDWaveLayerAdvanced
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.adaptation, pyfolds.backprop, pyfolds.inhibition, pyfolds.refractory, pyfolds.short_term, pyfolds.stdp

### `pyfolds.advanced.adaptation`
- Arquivo: `advanced/adaptation.py`
- Classes: AdaptationMixin
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.utils.types

### `pyfolds.advanced.backprop`
- Arquivo: `advanced/backprop.py`
- Classes: BackpropMixin
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.advanced.time_mixin

### `pyfolds.advanced.inhibition`
- Arquivo: `advanced/inhibition.py`
- Classes: InhibitionLayer, InhibitionMixin
- Funções de topo: (nenhuma)
- Imports internos: (nenhum)

### `pyfolds.advanced.refractory`
- Arquivo: `advanced/refractory.py`
- Classes: RefractoryMixin
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.advanced.time_mixin

### `pyfolds.advanced.short_term`
- Arquivo: `advanced/short_term.py`
- Classes: ShortTermDynamicsMixin
- Funções de topo: (nenhuma)
- Imports internos: (nenhum)

### `pyfolds.advanced.stdp`
- Arquivo: `advanced/stdp.py`
- Classes: STDPMixin
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.utils.types

### `pyfolds.advanced.time_mixin`
- Arquivo: `advanced/time_mixin.py`
- Classes: TimedMixin
- Funções de topo: (nenhuma)
- Imports internos: (nenhum)

### `pyfolds.core`
- Arquivo: `core/__init__.py`
- Classes: (nenhuma)
- Funções de topo: create_accumulator, create_neuron, create_neuron_v2, demo
- Imports internos: pyfolds.accumulator, pyfolds.base, pyfolds.config, pyfolds.dendrite, pyfolds.factory, pyfolds.homeostasis, pyfolds.neuromodulation, pyfolds.neuron, pyfolds.neuron_v2, pyfolds.synapse

### `pyfolds.core.accumulator`
- Arquivo: `core/accumulator.py`
- Classes: AccumulatedStats, StatisticsAccumulator
- Funções de topo: create_accumulator_from_config
- Imports internos: (nenhum)

### `pyfolds.core.base`
- Arquivo: `core/base.py`
- Classes: BasePlasticityRule, BaseNeuron
- Funções de topo: (nenhuma)
- Imports internos: (nenhum)

### `pyfolds.core.config`
- Arquivo: `core/config.py`
- Classes: MPJRDConfig
- Funções de topo: (nenhuma)
- Imports internos: (nenhum)

### `pyfolds.core.dendrite`
- Arquivo: `core/dendrite.py`
- Classes: MPJRDDendrite
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.core.config, pyfolds.core.synapse

### `pyfolds.core.factory`
- Arquivo: `core/factory.py`
- Classes: NeuronType, NeuronFactory
- Funções de topo: get_available_types, infer_neuron_type, register_default_neurons, register_neuron
- Imports internos: pyfolds.core.config

### `pyfolds.core.homeostasis`
- Arquivo: `core/homeostasis.py`
- Classes: HomeostasisController
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.core.config

### `pyfolds.core.neuromodulation`
- Arquivo: `core/neuromodulation.py`
- Classes: Neuromodulator
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.core.config

### `pyfolds.core.neuron`
- Arquivo: `core/neuron.py`
- Classes: MPJRDNeuron
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.core.accumulator, pyfolds.core.base, pyfolds.core.config, pyfolds.core.dendrite, pyfolds.core.homeostasis, pyfolds.core.neuromodulation, pyfolds.utils.logging, pyfolds.utils.types, pyfolds.utils.validation

### `pyfolds.core.neuron_v2`
- Arquivo: `core/neuron_v2.py`
- Classes: MPJRDNeuronV2
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.core.neuron, pyfolds.utils.types

### `pyfolds.core.synapse`
- Arquivo: `core/synapse.py`
- Classes: MPJRDSynapse
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.core.config, pyfolds.utils.math, pyfolds.utils.types

### `pyfolds.factory`
- Arquivo: `factory.py`
- Classes: NeuronType, NeuronFactory
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.core.base, pyfolds.core.config, pyfolds.core.neuron, pyfolds.wave.neuron

### `pyfolds.layers`
- Arquivo: `layers/__init__.py`
- Classes: (nenhuma)
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.layer, pyfolds.wave_layer

### `pyfolds.layers.layer`
- Arquivo: `layers/layer.py`
- Classes: MPJRDLayer
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.core.config, pyfolds.core.neuron, pyfolds.utils.types

### `pyfolds.layers.wave_layer`
- Arquivo: `layers/wave_layer.py`
- Classes: MPJRDWaveLayer
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.layers.layer, pyfolds.wave

### `pyfolds.monitoring`
- Arquivo: `monitoring/__init__.py`
- Classes: (nenhuma)
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.health

### `pyfolds.monitoring.health`
- Arquivo: `monitoring/health.py`
- Classes: HealthStatus, NeuronHealthCheck, NeuronHealthMonitor
- Funções de topo: (nenhuma)
- Imports internos: (nenhum)

### `pyfolds.network`
- Arquivo: `network/__init__.py`
- Classes: (nenhuma)
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.builder, pyfolds.network, pyfolds.wave_network

### `pyfolds.network.builder`
- Arquivo: `network/builder.py`
- Classes: NetworkBuilder
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.core.config, pyfolds.layers.layer, pyfolds.network.network

### `pyfolds.network.network`
- Arquivo: `network/network.py`
- Classes: MPJRDNetwork
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.layers, pyfolds.utils.types

### `pyfolds.network.wave_network`
- Arquivo: `network/wave_network.py`
- Classes: MPJRDWaveNetwork
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.layers, pyfolds.network.network, pyfolds.wave

### `pyfolds.serialization`
- Arquivo: `serialization/__init__.py`
- Classes: (nenhuma)
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.ecc, pyfolds.foldio, pyfolds.versioned_checkpoint

### `pyfolds.serialization.ecc`
- Arquivo: `serialization/ecc.py`
- Classes: ECCResult, ECCCodec, NoECC, ReedSolomonECC
- Funções de topo: ecc_from_protection
- Imports internos: (nenhum)

### `pyfolds.serialization.foldio`
- Arquivo: `serialization/foldio.py`
- Classes: FoldSecurityError, FoldWriter, FoldReader, _TrustedFoldReader
- Funções de topo: _build_nuclear_npz, _cfg_to_dict, _crc32c_fallback, _expression_summary, _history_snapshot, _init_crc32c_table, _json_bytes, _optional_import, _reproducibility_metadata, _safe_git_hash, _telemetry_snapshot, crc32c_u32, is_mind, is_mind_chunks, load_fold_or_mind, peek_fold_or_mind, peek_mind, read_nuclear_arrays, save_fold_or_mind, sha256_hex
- Imports internos: pyfolds.serialization.ecc

### `pyfolds.serialization.versioned_checkpoint`
- Arquivo: `serialization/versioned_checkpoint.py`
- Classes: VersionedCheckpoint
- Funções de topo: (nenhuma)
- Imports internos: (nenhum)

### `pyfolds.telemetry`
- Arquivo: `telemetry/__init__.py`
- Classes: (nenhuma)
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.controller, pyfolds.decorator, pyfolds.events, pyfolds.ringbuffer, pyfolds.sinks, pyfolds.types

### `pyfolds.telemetry.controller`
- Arquivo: `telemetry/controller.py`
- Classes: TelemetryProfile, TelemetryStats, TelemetryConfig, TelemetryController
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.telemetry.events, pyfolds.telemetry.ringbuffer, pyfolds.telemetry.sinks

### `pyfolds.telemetry.decorator`
- Arquivo: `telemetry/decorator.py`
- Classes: (nenhuma)
- Funções de topo: telemetry
- Imports internos: pyfolds.telemetry.controller, pyfolds.telemetry.events

### `pyfolds.telemetry.events`
- Arquivo: `telemetry/events.py`
- Classes: TelemetryEvent
- Funções de topo: commit_event, commit_event_lazy, forward_event, forward_event_lazy, sleep_event, sleep_event_lazy
- Imports internos: (nenhum)

### `pyfolds.telemetry.ringbuffer`
- Arquivo: `telemetry/ringbuffer.py`
- Classes: RingBuffer
- Funções de topo: (nenhuma)
- Imports internos: (nenhum)

### `pyfolds.telemetry.sinks`
- Arquivo: `telemetry/sinks.py`
- Classes: Sink, NoOpSink, MemorySink, ConsoleSink, JSONLinesSink, DistributorSink
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.telemetry.events, pyfolds.telemetry.ringbuffer

### `pyfolds.telemetry.types`
- Arquivo: `telemetry/types.py`
- Classes: ForwardPayload, CommitPayload, SleepPayload
- Funções de topo: (nenhuma)
- Imports internos: (nenhum)

### `pyfolds.utils`
- Arquivo: `utils/__init__.py`
- Classes: (nenhuma)
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.device, pyfolds.logging, pyfolds.math, pyfolds.types, pyfolds.validation

### `pyfolds.utils.context`
- Arquivo: `utils/context.py`
- Classes: (nenhuma)
- Funções de topo: learning_mode
- Imports internos: pyfolds.core.base, pyfolds.utils.types

### `pyfolds.utils.device`
- Arquivo: `utils/device.py`
- Classes: DeviceManager
- Funções de topo: ensure_device, get_device, infer_device
- Imports internos: (nenhum)

### `pyfolds.utils.logging`
- Arquivo: `utils/logging.py`
- Classes: StructuredFormatter, PyFoldsLogger
- Funções de topo: get_logger, setup_logging, trace
- Imports internos: (nenhum)

### `pyfolds.utils.math`
- Arquivo: `utils/math.py`
- Classes: (nenhuma)
- Funções de topo: calculate_vc_dimension, clamp_R, clamp_rate, safe_div, safe_weight_law, xavier_init
- Imports internos: (nenhum)

### `pyfolds.utils.types`
- Arquivo: `utils/types.py`
- Classes: LearningMode, ConnectionType, ModeConfig, AdaptationOutput, AdaptationConfig
- Funções de topo: (nenhuma)
- Imports internos: (nenhum)

### `pyfolds.utils.validation`
- Arquivo: `utils/validation.py`
- Classes: (nenhuma)
- Funções de topo: validate_device_consistency, validate_input
- Imports internos: (nenhum)

### `pyfolds.wave`
- Arquivo: `wave/__init__.py`
- Classes: (nenhuma)
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.config, pyfolds.neuron

### `pyfolds.wave.config`
- Arquivo: `wave/config.py`
- Classes: MPJRDWaveConfig
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.core.config

### `pyfolds.wave.neuron`
- Arquivo: `wave/neuron.py`
- Classes: MPJRDWaveNeuron
- Funções de topo: (nenhuma)
- Imports internos: pyfolds.core.neuron, pyfolds.utils.types, pyfolds.utils.validation, pyfolds.wave.config
