# Code Map (Sheer Audit)

- Repositório: `pyfolds`
- Arquivos Python: `143`
- Símbolos: `1081`

## Módulos

### `src.pyfolds`
- Arquivo: `src/pyfolds/__init__.py`
- Imports:
  - `mod:src.advanced`
  - `mod:src.bridge`
  - `mod:src.core.base`
  - `mod:src.core.config`
  - `mod:src.core.factory`
  - `mod:src.core.neuron`
  - `mod:src.core.neuron_v2`
  - `mod:src.layers`
  - `mod:src.monitoring`
  - `mod:src.network`
  - `mod:src.serialization`
  - `mod:src.telemetry`
  - `mod:src.utils.context`
  - `mod:src.utils.types`
  - `mod:src.wave`
  - `mod:typing`
  - `mod:warnings`
- Funções:
  - `__getattr__(name)`

### `src.pyfolds.advanced`
- Arquivo: `src/pyfolds/advanced/__init__.py`
- Imports:
  - `mod:logging`
  - `mod:src.core.neuron`
  - `mod:src.layers.layer`
  - `mod:src.pyfolds.adaptation`
  - `mod:src.pyfolds.backprop`
  - `mod:src.pyfolds.inhibition`
  - `mod:src.pyfolds.refractory`
  - `mod:src.pyfolds.short_term`
  - `mod:src.pyfolds.stdp`
  - `mod:src.utils.logging`
  - `mod:src.wave`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `MPJRDLayerAdvanced` (bases: MPJRDLayer)
  - `MPJRDNeuronAdvanced` (bases: BackpropMixin, ShortTermDynamicsMixin, STDPMixin, AdaptationMixin, RefractoryMixin, MPJRDNeuronBase)
  - `MPJRDWaveLayerAdvanced` (bases: MPJRDLayer)
  - `MPJRDWaveNeuronAdvanced` (bases: BackpropMixin, ShortTermDynamicsMixin, STDPMixin, AdaptationMixin, RefractoryMixin, MPJRDWaveNeuronBase)

### `src.pyfolds.advanced.adaptation`
- Arquivo: `src/pyfolds/advanced/adaptation.py`
- Imports:
  - `mod:math`
  - `mod:src.pyfolds.utils.types`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `AdaptationMixin` (bases: (sem base explícita))

### `src.pyfolds.advanced.backprop`
- Arquivo: `src/pyfolds/advanced/backprop.py`
- Imports:
  - `mod:collections`
  - `mod:math`
  - `mod:src.pyfolds.advanced.time_mixin`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `BackpropMixin` (bases: TimedMixin)

### `src.pyfolds.advanced.inhibition`
- Arquivo: `src/pyfolds/advanced/inhibition.py`
- Imports:
  - `mod:logging`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:typing`
- Classes:
  - `InhibitionLayer` (bases: nn.Module)
  - `InhibitionMixin` (bases: (sem base explícita))

### `src.pyfolds.advanced.refractory`
- Arquivo: `src/pyfolds/advanced/refractory.py`
- Imports:
  - `mod:src.pyfolds.advanced.time_mixin`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `RefractoryMixin` (bases: TimedMixin)

### `src.pyfolds.advanced.short_term`
- Arquivo: `src/pyfolds/advanced/short_term.py`
- Imports:
  - `mod:math`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `ShortTermDynamicsMixin` (bases: (sem base explícita))

### `src.pyfolds.advanced.stdp`
- Arquivo: `src/pyfolds/advanced/stdp.py`
- Imports:
  - `mod:math`
  - `mod:src.pyfolds.utils.types`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `STDPMixin` (bases: (sem base explícita))

### `src.pyfolds.advanced.time_mixin`
- Arquivo: `src/pyfolds/advanced/time_mixin.py`
- Imports:
  - `mod:torch`
- Classes:
  - `TimedMixin` (bases: (sem base explícita))

### `src.pyfolds.bridge`
- Arquivo: `src/pyfolds/bridge/__init__.py`
- Imports:
  - `mod:src.pyfolds.dispatcher`

### `src.pyfolds.bridge.dispatcher`
- Arquivo: `src/pyfolds/bridge/dispatcher.py`
- Imports:
  - `mod:__future__`
  - `mod:datetime`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `MindDispatcher` (bases: (sem base explícita))

### `src.pyfolds.contracts`
- Arquivo: `src/pyfolds/contracts/__init__.py`
- Imports:
  - `mod:src.pyfolds.backends`
  - `mod:src.pyfolds.neuron_contract`

### `src.pyfolds.contracts.backends`
- Arquivo: `src/pyfolds/contracts/backends.py`
- Imports:
  - `mod:__future__`
  - `mod:dataclasses`
  - `mod:numpy`
  - `mod:src.pyfolds.contracts.neuron_contract`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `TensorFlowNeuronContractBackend` (bases: _BaseContractBackend)
  - `TorchNeuronContractBackend` (bases: _BaseContractBackend)
  - `_BaseContractBackend` (bases: (sem base explícita))
  - `_State` (bases: (sem base explícita))

### `src.pyfolds.contracts.neuron_contract`
- Arquivo: `src/pyfolds/contracts/neuron_contract.py`
- Imports:
  - `mod:__future__`
  - `mod:dataclasses`
  - `mod:enum`
  - `mod:typing`
- Classes:
  - `ContractViolation` (bases: ValueError)
  - `MechanismStep` (bases: str, Enum)
  - `NeuronStepInput` (bases: (sem base explícita))
  - `NeuronStepOutput` (bases: (sem base explícita))
  - `StepExecutionTrace` (bases: (sem base explícita))
- Funções:
  - `validate_step_output(output, dt)`

### `src.pyfolds.core`
- Arquivo: `src/pyfolds/core/__init__.py`
- Imports:
  - `mod:src.pyfolds.accumulator`
  - `mod:src.pyfolds.base`
  - `mod:src.pyfolds.config`
  - `mod:src.pyfolds.dendrite`
  - `mod:src.pyfolds.dendrite_integration`
  - `mod:src.pyfolds.factory`
  - `mod:src.pyfolds.homeostasis`
  - `mod:src.pyfolds.neuromodulation`
  - `mod:src.pyfolds.neuron`
  - `mod:src.pyfolds.neuron_v2`
  - `mod:src.pyfolds.synapse`
- Funções:
  - `create_accumulator(cfg, track_extra)`
  - `create_neuron(cfg)`
  - `create_neuron_v2(cfg)`
  - `demo()`

### `src.pyfolds.core.accumulator`
- Arquivo: `src/pyfolds/core/accumulator.py`
- Imports:
  - `mod:collections`
  - `mod:dataclasses`
  - `mod:threading`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:typing`
- Classes:
  - `AccumulatedStats` (bases: (sem base explícita))
  - `StatisticsAccumulator` (bases: nn.Module)
- Funções:
  - `create_accumulator_from_config(config, track_extra)`

### `src.pyfolds.core.base`
- Arquivo: `src/pyfolds/core/base.py`
- Imports:
  - `mod:__future__`
  - `mod:abc`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:typing`
- Classes:
  - `BaseNeuron` (bases: nn.Module, ABC)
  - `BasePlasticityRule` (bases: ABC)

### `src.pyfolds.core.config`
- Arquivo: `src/pyfolds/core/config.py`
- Imports:
  - `mod:dataclasses`
  - `mod:math`
  - `mod:typing`
  - `mod:warnings`
- Classes:
  - `MPJRDConfig` (bases: (sem base explícita))

### `src.pyfolds.core.dendrite`
- Arquivo: `src/pyfolds/core/dendrite.py`
- Imports:
  - `mod:logging`
  - `mod:src.pyfolds.core.config`
  - `mod:src.pyfolds.core.synapse`
  - `mod:threading`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:typing`
- Classes:
  - `MPJRDDendrite` (bases: nn.Module)

### `src.pyfolds.core.dendrite_integration`
- Arquivo: `src/pyfolds/core/dendrite_integration.py`
- Imports:
  - `mod:__future__`
  - `mod:src.pyfolds.core.config`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:typing`
- Classes:
  - `DendriticIntegration` (bases: nn.Module)
  - `DendriticOutput` (bases: NamedTuple)

### `src.pyfolds.core.factory`
- Arquivo: `src/pyfolds/core/factory.py`
- Imports:
  - `mod:__future__`
  - `mod:enum`
  - `mod:src.pyfolds.core.config`
  - `mod:typing`
  - `mod:warnings`
- Classes:
  - `NeuronFactory` (bases: (sem base explícita))
  - `NeuronType` (bases: Enum)
- Funções:
  - `get_available_types()`
  - `infer_neuron_type(cfg)`
  - `register_default_neurons()`
  - `register_neuron(neuron_type)`

### `src.pyfolds.core.homeostasis`
- Arquivo: `src/pyfolds/core/homeostasis.py`
- Imports:
  - `mod:logging`
  - `mod:math`
  - `mod:src.pyfolds.core.config`
  - `mod:threading`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:typing`
- Classes:
  - `HomeostasisController` (bases: nn.Module)

### `src.pyfolds.core.neuromodulation`
- Arquivo: `src/pyfolds/core/neuromodulation.py`
- Imports:
  - `mod:math`
  - `mod:src.pyfolds.core.config`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:typing`
- Classes:
  - `Neuromodulator` (bases: nn.Module)

### `src.pyfolds.core.neuron`
- Arquivo: `src/pyfolds/core/neuron.py`
- Imports:
  - `mod:dataclasses`
  - `mod:queue`
  - `mod:src.pyfolds.core.accumulator`
  - `mod:src.pyfolds.core.base`
  - `mod:src.pyfolds.core.config`
  - `mod:src.pyfolds.core.dendrite`
  - `mod:src.pyfolds.core.dendrite_integration`
  - `mod:src.pyfolds.core.homeostasis`
  - `mod:src.pyfolds.core.neuromodulation`
  - `mod:src.pyfolds.utils.logging`
  - `mod:src.pyfolds.utils.types`
  - `mod:src.pyfolds.utils.validation`
  - `mod:threading`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:typing`
- Classes:
  - `MPJRDNeuron` (bases: BaseNeuron)

### `src.pyfolds.core.neuron_v2`
- Arquivo: `src/pyfolds/core/neuron_v2.py`
- Imports:
  - `mod:src.pyfolds.core.neuron`
  - `mod:src.pyfolds.utils.types`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `MPJRDNeuronV2` (bases: MPJRDNeuron)

### `src.pyfolds.core.synapse`
- Arquivo: `src/pyfolds/core/synapse.py`
- Imports:
  - `mod:src.pyfolds.core.config`
  - `mod:src.pyfolds.utils.math`
  - `mod:src.pyfolds.utils.types`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:typing`
- Classes:
  - `MPJRDSynapse` (bases: nn.Module)

### `src.pyfolds.factory`
- Arquivo: `src/pyfolds/factory.py`
- Imports:
  - `mod:__future__`
  - `mod:enum`
  - `mod:src.pyfolds.core.base`
  - `mod:src.pyfolds.core.config`
  - `mod:src.pyfolds.core.neuron`
  - `mod:src.pyfolds.wave.neuron`
  - `mod:typing`
- Classes:
  - `NeuronFactory` (bases: (sem base explícita))
  - `NeuronType` (bases: str, Enum)

### `src.pyfolds.layers`
- Arquivo: `src/pyfolds/layers/__init__.py`
- Imports:
  - `mod:src.pyfolds.layer`
  - `mod:src.pyfolds.wave_layer`

### `src.pyfolds.layers.layer`
- Arquivo: `src/pyfolds/layers/layer.py`
- Imports:
  - `mod:contextlib`
  - `mod:src.pyfolds.core.config`
  - `mod:src.pyfolds.core.neuron`
  - `mod:src.pyfolds.utils.types`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:typing`
  - `mod:warnings`
- Classes:
  - `MPJRDLayer` (bases: nn.Module)

### `src.pyfolds.layers.wave_layer`
- Arquivo: `src/pyfolds/layers/wave_layer.py`
- Imports:
  - `mod:src.pyfolds.layers.layer`
  - `mod:src.pyfolds.wave`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `MPJRDWaveLayer` (bases: MPJRDLayer)

### `src.pyfolds.monitoring`
- Arquivo: `src/pyfolds/monitoring/__init__.py`
- Imports:
  - `mod:src.pyfolds.health`
  - `mod:src.pyfolds.mindcontrol`

### `src.pyfolds.monitoring.health`
- Arquivo: `src/pyfolds/monitoring/health.py`
- Imports:
  - `mod:__future__`
  - `mod:enum`
  - `mod:hashlib`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `HealthStatus` (bases: Enum)
  - `ModelIntegrityMonitor` (bases: WeightIntegrityMonitor)
  - `NeuronHealthCheck` (bases: (sem base explícita))
  - `NeuronHealthMonitor` (bases: (sem base explícita))
  - `WeightIntegrityMonitor` (bases: (sem base explícita))

### `src.pyfolds.monitoring.mindcontrol`
- Arquivo: `src/pyfolds/monitoring/mindcontrol.py`
- Imports:
  - `mod:__future__`
  - `mod:dataclasses`
  - `mod:queue`
  - `mod:src.pyfolds.telemetry.events`
  - `mod:src.pyfolds.telemetry.sinks`
  - `mod:threading`
  - `mod:typing`
  - `mod:weakref`
- Classes:
  - `MindControl` (bases: (sem base explícita))
  - `MindControlEngine` (bases: (sem base explícita))
  - `MindControlSink` (bases: Sink)
  - `MutationCommand` (bases: (sem base explícita))
  - `MutationQueue` (bases: (sem base explícita))

### `src.pyfolds.network`
- Arquivo: `src/pyfolds/network/__init__.py`
- Imports:
  - `mod:src.pyfolds.builder`
  - `mod:src.pyfolds.network`
  - `mod:src.pyfolds.wave_network`

### `src.pyfolds.network.builder`
- Arquivo: `src/pyfolds/network/builder.py`
- Imports:
  - `mod:__future__`
  - `mod:src.pyfolds.core.config`
  - `mod:src.pyfolds.layers.layer`
  - `mod:src.pyfolds.network.network`
  - `mod:typing`
- Classes:
  - `NetworkBuilder` (bases: (sem base explícita))

### `src.pyfolds.network.network`
- Arquivo: `src/pyfolds/network/network.py`
- Imports:
  - `mod:collections`
  - `mod:src.pyfolds.layers`
  - `mod:src.pyfolds.utils.types`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:typing`
- Classes:
  - `MPJRDNetwork` (bases: nn.Module)

### `src.pyfolds.network.wave_network`
- Arquivo: `src/pyfolds/network/wave_network.py`
- Imports:
  - `mod:src.pyfolds.layers`
  - `mod:src.pyfolds.network.network`
  - `mod:src.pyfolds.wave`
- Classes:
  - `MPJRDWaveNetwork` (bases: MPJRDNetwork)

### `src.pyfolds.serialization`
- Arquivo: `src/pyfolds/serialization/__init__.py`
- Imports:
  - `mod:src.pyfolds.ecc`
  - `mod:src.pyfolds.foldio`
  - `mod:src.pyfolds.versioned_checkpoint`

### `src.pyfolds.serialization.ecc`
- Arquivo: `src/pyfolds/serialization/ecc.py`
- Imports:
  - `mod:__future__`
  - `mod:dataclasses`
  - `mod:typing`
- Classes:
  - `ECCCodec` (bases: Protocol)
  - `ECCProtector` (bases: (sem base explícita))
  - `ECCResult` (bases: (sem base explícita))
  - `NoECC` (bases: (sem base explícita))
  - `ReedSolomonECC` (bases: (sem base explícita))
- Funções:
  - `ecc_from_protection(level)`

### `src.pyfolds.serialization.foldio`
- Arquivo: `src/pyfolds/serialization/foldio.py`
- Imports:
  - `mod:__future__`
  - `mod:dataclasses`
  - `mod:hashlib`
  - `mod:hmac`
  - `mod:importlib`
  - `mod:io`
  - `mod:json`
  - `mod:mmap`
  - `mod:numpy`
  - `mod:os`
  - `mod:pathlib`
  - `mod:platform`
  - `mod:shutil`
  - `mod:src.pyfolds.serialization.ecc`
  - `mod:struct`
  - `mod:subprocess`
  - `mod:sys`
  - `mod:time`
  - `mod:torch`
  - `mod:typing`
  - `mod:warnings`
- Classes:
  - `FoldReader` (bases: (sem base explícita))
  - `FoldSecurityError` (bases: RuntimeError)
  - `FoldWriter` (bases: (sem base explícita))
  - `_TrustedFoldReader` (bases: FoldReader)
- Funções:
  - `_build_nuclear_npz(neuron)`
  - `_canonical_json(obj)`
  - `_cfg_to_dict(cfg)`
  - `_collect_hyperparameters(neuron)`
  - `_crc32c_fallback(data)`
  - `_deserialize_state_dict_safetensors(payload, map_location)`
  - `_expression_summary(neuron)`
  - `_history_snapshot(neuron)`
  - `_init_crc32c_table()`
  - `_json_bytes(obj)`
  - `_optional_import(module_name)`
  - `_reproducibility_metadata()`
  - `_safe_git_hash()`
  - `_serialize_state_dict_safetensors(state_dict)`
  - `_sign_payload_ed25519(payload, private_key_pem)`
  - `_telemetry_snapshot(neuron, max_events)`
  - `_validate_safetensors_payload(payload)`
  - `_verify_payload_signature_ed25519(payload, signature_hex, public_key_pem)`
  - `crc32c_u32(data)`
  - `is_mind(path)`
  - `is_mind_chunks(chunks)`
  - `load_fold_or_mind(path, neuron_class, map_location, trusted_torch_payload, signature_public_key_pem)`
  - `peek_fold_or_mind(path, use_mmap)`
  - `peek_mind(path, use_mmap)`
  - `read_nuclear_arrays(path, use_mmap, verify)`
  - `save_fold_or_mind(neuron, path, tags, include_history, include_telemetry, include_nuclear_arrays, compress, ecc, protection, extra_manifest, dataset_manifest, performance_manifest, fairness_manifest, explainability_manifest, compliance_manifest, audit_events, signature_private_key_pem, signature_key_id)`
  - `sha256_hex(data)`

### `src.pyfolds.serialization.versioned_checkpoint`
- Arquivo: `src/pyfolds/serialization/versioned_checkpoint.py`
- Imports:
  - `mod:__future__`
  - `mod:dataclasses`
  - `mod:datetime`
  - `mod:hashlib`
  - `mod:hmac`
  - `mod:json`
  - `mod:os`
  - `mod:pathlib`
  - `mod:pickle`
  - `mod:pyfolds.core.config`
  - `mod:shutil`
  - `mod:src.pyfolds.serialization.ecc`
  - `mod:subprocess`
  - `mod:torch`
  - `mod:typing`
  - `mod:warnings`
- Classes:
  - `VersionedCheckpoint` (bases: (sem base explícita))

### `src.pyfolds.telemetry`
- Arquivo: `src/pyfolds/telemetry/__init__.py`
- Imports:
  - `mod:src.pyfolds.controller`
  - `mod:src.pyfolds.decorator`
  - `mod:src.pyfolds.events`
  - `mod:src.pyfolds.ringbuffer`
  - `mod:src.pyfolds.sinks`
  - `mod:src.pyfolds.types`

### `src.pyfolds.telemetry.controller`
- Arquivo: `src/pyfolds/telemetry/controller.py`
- Imports:
  - `mod:dataclasses`
  - `mod:enum`
  - `mod:logging`
  - `mod:random`
  - `mod:src.pyfolds.telemetry.events`
  - `mod:src.pyfolds.telemetry.ringbuffer`
  - `mod:src.pyfolds.telemetry.sinks`
  - `mod:threading`
  - `mod:typing`
- Classes:
  - `TelemetryConfig` (bases: (sem base explícita))
  - `TelemetryController` (bases: (sem base explícita))
  - `TelemetryProfile` (bases: str, Enum)
  - `TelemetryStats` (bases: TypedDict)

### `src.pyfolds.telemetry.decorator`
- Arquivo: `src/pyfolds/telemetry/decorator.py`
- Imports:
  - `mod:functools`
  - `mod:src.pyfolds.telemetry.controller`
  - `mod:src.pyfolds.telemetry.events`
  - `mod:time`
  - `mod:typing`
- Funções:
  - `telemetry(phase, sample_rate, capture_args, capture_return)`

### `src.pyfolds.telemetry.events`
- Arquivo: `src/pyfolds/telemetry/events.py`
- Imports:
  - `mod:dataclasses`
  - `mod:time`
  - `mod:typing`
- Classes:
  - `TelemetryEvent` (bases: (sem base explícita))
- Funções:
  - `commit_event(step_id, mode, neuron_id)`
  - `commit_event_lazy(step_id, mode, payload_fn, neuron_id)`
  - `forward_event(step_id, mode, neuron_id)`
  - `forward_event_lazy(step_id, mode, payload_fn, neuron_id)`
  - `sleep_event(step_id, mode, neuron_id)`
  - `sleep_event_lazy(step_id, mode, payload_fn, neuron_id)`

### `src.pyfolds.telemetry.ringbuffer`
- Arquivo: `src/pyfolds/telemetry/ringbuffer.py`
- Imports:
  - `mod:collections.abc`
  - `mod:threading`
  - `mod:typing`
- Classes:
  - `RingBuffer` (bases: Generic[T])

### `src.pyfolds.telemetry.sinks`
- Arquivo: `src/pyfolds/telemetry/sinks.py`
- Imports:
  - `mod:abc`
  - `mod:json`
  - `mod:logging`
  - `mod:pathlib`
  - `mod:src.pyfolds.telemetry.events`
  - `mod:src.pyfolds.telemetry.ringbuffer`
  - `mod:typing`
- Classes:
  - `BufferedJSONLinesSink` (bases: JSONLinesSink)
  - `ConsoleSink` (bases: Sink)
  - `DistributorSink` (bases: Sink)
  - `JSONLinesSink` (bases: Sink)
  - `MemorySink` (bases: Sink)
  - `NoOpSink` (bases: Sink)
  - `Sink` (bases: ABC)

### `src.pyfolds.telemetry.types`
- Arquivo: `src/pyfolds/telemetry/types.py`
- Imports:
  - `mod:typing`
- Classes:
  - `CommitPayload` (bases: TypedDict)
  - `ForwardPayload` (bases: TypedDict)
  - `SleepPayload` (bases: TypedDict)

### `src.pyfolds.tf`
- Arquivo: `src/pyfolds/tf/__init__.py`
- Imports:
  - `mod:__future__`
  - `mod:importlib.util`
- Funções:
  - `__getattr__(name)`

### `src.pyfolds.tf.layers`
- Arquivo: `src/pyfolds/tf/layers.py`
- Imports:
  - `mod:__future__`
  - `mod:src.pyfolds.tf.neuron`
- Classes:
  - `MPJRDTFLayer` (bases: tf.keras.layers.Layer)

### `src.pyfolds.tf.neuron`
- Arquivo: `src/pyfolds/tf/neuron.py`
- Imports:
  - `mod:__future__`
  - `mod:typing`
- Classes:
  - `MPJRDTFNeuronCell` (bases: tf.keras.layers.Layer)

### `src.pyfolds.utils`
- Arquivo: `src/pyfolds/utils/__init__.py`
- Imports:
  - `mod:src.pyfolds.device`
  - `mod:src.pyfolds.logging`
  - `mod:src.pyfolds.math`
  - `mod:src.pyfolds.types`
  - `mod:src.pyfolds.validation`

### `src.pyfolds.utils.context`
- Arquivo: `src/pyfolds/utils/context.py`
- Imports:
  - `mod:__future__`
  - `mod:contextlib`
  - `mod:src.pyfolds.core.base`
  - `mod:src.pyfolds.utils.types`
  - `mod:typing`
- Funções:
  - `learning_mode(neuron, mode)`

### `src.pyfolds.utils.device`
- Arquivo: `src/pyfolds/utils/device.py`
- Imports:
  - `mod:logging`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `DeviceManager` (bases: (sem base explícita))
- Funções:
  - `ensure_device(tensor, device)`
  - `get_device(device)`
  - `infer_device(inputs)`

### `src.pyfolds.utils.logging`
- Arquivo: `src/pyfolds/utils/logging.py`
- Imports:
  - `mod:collections`
  - `mod:datetime`
  - `mod:json`
  - `mod:logging`
  - `mod:logging.handlers`
  - `mod:pathlib`
  - `mod:sys`
  - `mod:threading`
  - `mod:typing`
- Classes:
  - `CircularBufferFileHandler` (bases: logging.Handler)
  - `FixedLayoutFormatter` (bases: logging.Formatter)
  - `PyFoldsLogger` (bases: (sem base explícita))
  - `StructuredFormatter` (bases: logging.Formatter)
- Funções:
  - `build_log_path(log_dir, app, version)`
  - `get_logger(name)`
  - `next_log_path(log_dir, app, version)`
  - `setup_logging(log_file, level, structured, circular_buffer_lines, circular_buffer_flush_interval_sec, console, fixed_layout)`
  - `setup_run_logging(app, version, log_dir, level, structured, fixed_layout, console, circular_buffer_lines, circular_buffer_flush_interval_sec)`
  - `trace(self, message)`

### `src.pyfolds.utils.math`
- Arquivo: `src/pyfolds/utils/math.py`
- Imports:
  - `mod:math`
  - `mod:torch`
  - `mod:typing`
- Funções:
  - `calculate_vc_dimension(n_neurons, n_dendrites, n_synapses, avg_connections)`
  - `clamp_R(r)`
  - `clamp_rate(r)`
  - `safe_div(x, y, eps)`
  - `safe_weight_law(N, w_scale, max_log_val, enforce_checks)`
  - `xavier_init(shape, gain)`

### `src.pyfolds.utils.types`
- Arquivo: `src/pyfolds/utils/types.py`
- Imports:
  - `mod:dataclasses`
  - `mod:enum`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `AdaptationConfig` (bases: (sem base explícita))
  - `AdaptationOutput` (bases: NamedTuple)
  - `ConnectionType` (bases: Enum)
  - `LearningMode` (bases: Enum)
  - `ModeConfig` (bases: (sem base explícita))
- Funções:
  - `normalize_learning_mode(mode)`

### `src.pyfolds.utils.validation`
- Arquivo: `src/pyfolds/utils/validation.py`
- Imports:
  - `mod:__future__`
  - `mod:functools`
  - `mod:torch`
  - `mod:typing`
- Funções:
  - `validate_device_consistency()`
  - `validate_input()`

### `src.pyfolds.wave`
- Arquivo: `src/pyfolds/wave/__init__.py`
- Imports:
  - `mod:src.pyfolds.config`
  - `mod:src.pyfolds.neuron`

### `src.pyfolds.wave.config`
- Arquivo: `src/pyfolds/wave/config.py`
- Imports:
  - `mod:dataclasses`
  - `mod:src.pyfolds.core.config`
  - `mod:typing`
- Classes:
  - `MPJRDWaveConfig` (bases: MPJRDConfig)

### `src.pyfolds.wave.neuron`
- Arquivo: `src/pyfolds/wave/neuron.py`
- Imports:
  - `mod:__future__`
  - `mod:src.pyfolds.core.neuron`
  - `mod:src.pyfolds.utils.types`
  - `mod:src.pyfolds.utils.validation`
  - `mod:src.pyfolds.wave.config`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `MPJRDWaveNeuron` (bases: MPJRDNeuron)

### `tests`
- Arquivo: `tests/__init__.py`

### `tests.checkpoint`
- Arquivo: `tests/checkpoint.py`
- Imports:
  - `mod:json`
  - `mod:logging`
  - `mod:pathlib`
  - `mod:time`
  - `mod:torch`
  - `mod:typing`
- Classes:
  - `CheckpointMixin` (bases: (sem base explícita))

### `tests.conftest`
- Arquivo: `tests/conftest.py`
- Imports:
  - `mod:__future__`
  - `mod:functools`
  - `mod:importlib.util`
  - `mod:pathlib`
  - `mod:pytest`
  - `mod:sys`
- Funções:
  - `_core_symbols()`
  - `_module_requires_torch(path)`
  - `batch_size()`
  - `device(torch_module)`
  - `full_config(_core_symbols)`
  - `pytest_collection_modifyitems(config, items)`
  - `pytest_configure(config)`
  - `pytest_report_header(config)`
  - `small_config(_core_symbols)`
  - `small_neuron(small_config, _core_symbols)`
  - `tiny_config(_core_symbols)`
  - `torch_module()`

### `tests.integration`
- Arquivo: `tests/integration/__init__.py`

### `tests.integration.test_advanced_integration`
- Arquivo: `tests/integration/test_advanced_integration.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.advanced`
  - `mod:torch`
- Funções:
  - `test_all_mixins_together_smoke()`

### `tests.integration.test_mindcontrol_runtime`
- Arquivo: `tests/integration/test_mindcontrol_runtime.py`
- Imports:
  - `mod:pyfolds.monitoring.mindcontrol`
  - `mod:pytest`
  - `mod:torch`
  - `mod:torch.nn`
- Classes:
  - `MockNeuron` (bases: nn.Module)
- Funções:
  - `test_mindcontrol_bounds_clamp_threshold_values()`
  - `test_mindcontrol_graph_safety()`

### `tests.integration.test_mnist_file_logging`
- Arquivo: `tests/integration/test_mnist_file_logging.py`
- Imports:
  - `mod:examples.mnist_file_logging`
  - `mod:pathlib`
  - `mod:pyfolds`
- Funções:
  - `test_pyfolds_imports_are_stable()`
  - `test_training_script_runs_end_to_end()`

### `tests.integration.test_neuron_advanced`
- Arquivo: `tests/integration/test_neuron_advanced.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestAdvancedNeuron` (bases: (sem base explícita))

### `tests.integration.test_temporal_sequence_stability`
- Arquivo: `tests/integration/test_temporal_sequence_stability.py`
- Imports:
  - `mod:importlib.util`
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Funções:
  - `forward_sequence(neuron, x_seq, dt)`
  - `test_forward_sequence_torch_stability_minimal_criteria()`
  - `test_tf_sequence_equivalent_stability_minimal_criteria()`

### `tests.integration.test_training_loop`
- Arquivo: `tests/integration/test_training_loop.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestTrainingLoop` (bases: (sem base explícita))

### `tests.performance`
- Arquivo: `tests/performance/__init__.py`

### `tests.performance.test_batch_speed`
- Arquivo: `tests/performance/test_batch_speed.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:time`
  - `mod:torch`
- Classes:
  - `TestBatchSpeed` (bases: (sem base explícita))

### `tests.performance.test_memory_usage`
- Arquivo: `tests/performance/test_memory_usage.py`
- Imports:
  - `mod:pyfolds`
  - `mod:torch`
  - `mod:tracemalloc`
- Classes:
  - `TestMemoryUsage` (bases: (sem base explícita))

### `tests.performance.test_telemetry_overhead`
- Arquivo: `tests/performance/test_telemetry_overhead.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:time`
  - `mod:torch`
- Funções:
  - `_measure_forward_loop(neuron, x)`
  - `_sync_if_cuda()`
  - `test_forward_telemetry_ringbuffer_overhead(small_config)`

### `tests.test_concurrent_reads`
- Arquivo: `tests/test_concurrent_reads.py`
- Imports:
  - `mod:concurrent.futures`
  - `mod:hashlib`
  - `mod:pyfolds`
  - `mod:pyfolds.core.neuron`
  - `mod:pyfolds.serialization`
  - `mod:pytest`
- Funções:
  - `_build_neuron()`
  - `_read_signature(path)`
  - `test_parallel_reads_are_consistent(tmp_path)`

### `tests.test_corruption_detection`
- Arquivo: `tests/test_corruption_detection.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.core.neuron`
  - `mod:pyfolds.core.synapse`
  - `mod:pyfolds.serialization`
  - `mod:pyfolds.serialization.foldio`
  - `mod:pytest`
  - `mod:struct`
  - `mod:torch`
- Funções:
  - `_build_neuron()`
  - `_write_base_fold(path)`
  - `test_bit_flip_is_detected(tmp_path)`
  - `test_ecc_like_burst_corruption_is_detected(tmp_path)`
  - `test_huge_index_len_dos_guard(tmp_path)`
  - `test_invalid_magic_is_rejected(tmp_path)`
  - `test_mpjrd_synapse_recovers_from_ecc_like_state_corruption()`
  - `test_partial_read_raises_eoferror(tmp_path)`
  - `test_truncation_is_detected(tmp_path)`

### `tests.test_fold_corruption`
- Arquivo: `tests/test_fold_corruption.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.core.neuron`
  - `mod:pyfolds.serialization`
  - `mod:pytest`
- Funções:
  - `test_fold_corruption_is_detected(tmp_path)`

### `tests.test_fold_fuzz`
- Arquivo: `tests/test_fold_fuzz.py`
- Imports:
  - `mod:os`
  - `mod:pyfolds.serialization`
  - `mod:pytest`
- Funções:
  - `test_fold_reader_rejects_random_payload(tmp_path)`

### `tests.test_stress_long`
- Arquivo: `tests/test_stress_long.py`
- Imports:
  - `mod:math`
  - `mod:pytest`
  - `mod:torch`
  - `mod:tracemalloc`
- Funções:
  - `test_100k_steps_stability_and_memory_signals(tmp_path)`

### `tests.tools.test_batch_create_issues`
- Arquivo: `tests/tools/test_batch_create_issues.py`
- Imports:
  - `mod:tools.batch_create_issues`
- Funções:
  - `test_validate_batch_structure_duplicate_ids()`

### `tests.tools.test_create_issue_report`
- Arquivo: `tests/tools/test_create_issue_report.py`
- Imports:
  - `mod:tools.create_issue_report`
- Funções:
  - `test_generate_yaml_frontmatter_contains_id()`

### `tests.tools.test_id_registry`
- Arquivo: `tests/tools/test_id_registry.py`
- Imports:
  - `mod:tools`
- Funções:
  - `test_next_adr_id_has_prefix()`
  - `test_next_issue_id_has_prefix()`

### `tests.tools.test_link_validation`
- Arquivo: `tests/tools/test_link_validation.py`
- Imports:
  - `mod:tools.validate_issue_format`
- Funções:
  - `test_validate_links_missing(tmp_path)`

### `tests.tools.test_sync_hub_auto`
- Arquivo: `tests/tools/test_sync_hub_auto.py`
- Imports:
  - `mod:tools.sync_hub_auto`
- Funções:
  - `test_status_to_emoji_done()`

### `tests.tools.test_validate_issue_format`
- Arquivo: `tests/tools/test_validate_issue_format.py`
- Imports:
  - `mod:pathlib`
  - `mod:tools.validate_issue_format`
- Funções:
  - `test_validate_structure_filename(tmp_path)`

### `tests.training.mnist_cifar_training_reference`
- Arquivo: `tests/training/mnist_cifar_training_reference.py`
- Imports:
  - `mod:__future__`
  - `mod:dataclasses`
  - `mod:pyfolds`
  - `mod:torch`
  - `mod:torch.nn`
  - `mod:torch.optim`
  - `mod:torch.utils.data`
  - `mod:torchvision`
  - `mod:typing`
- Classes:
  - `PyFoldsMLP` (bases: nn.Module)
  - `TrainingConfig` (bases: (sem base explícita))
- Funções:
  - `build_dataset(cfg)`
  - `evaluate(model, loader, device)`
  - `train_reference(cfg)`

### `tests.unit`
- Arquivo: `tests/unit/__init__.py`

### `tests.unit.advanced`
- Arquivo: `tests/unit/advanced/__init__.py`

### `tests.unit.advanced.test_adaptation`
- Arquivo: `tests/unit/advanced/test_adaptation.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestAdaptationMixin` (bases: (sem base explícita))

### `tests.unit.advanced.test_backprop`
- Arquivo: `tests/unit/advanced/test_backprop.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestBackpropMixin` (bases: (sem base explícita))

### `tests.unit.advanced.test_inhibition`
- Arquivo: `tests/unit/advanced/test_inhibition.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestInhibitionLayer` (bases: (sem base explícita))
  - `TestInhibitionMixin` (bases: (sem base explícita))

### `tests.unit.advanced.test_refractory`
- Arquivo: `tests/unit/advanced/test_refractory.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestRefractoryMixin` (bases: (sem base explícita))

### `tests.unit.advanced.test_short_term`
- Arquivo: `tests/unit/advanced/test_short_term.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestShortTermDynamicsMixin` (bases: (sem base explícita))

### `tests.unit.advanced.test_stdp`
- Arquivo: `tests/unit/advanced/test_stdp.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.utils.types`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestSTDPMixin` (bases: (sem base explícita))

### `tests.unit.bridge.test_dispatcher`
- Arquivo: `tests/unit/bridge/test_dispatcher.py`
- Imports:
  - `mod:pyfolds.bridge`
  - `mod:torch`
- Classes:
  - `_Cfg` (bases: (sem base explícita))
  - `_Layer` (bases: (sem base explícita))
  - `_Network` (bases: (sem base explícita))
- Funções:
  - `test_get_topology_map_uses_cfg_to_dict_when_available()`
  - `test_prepare_payload_tensor_fields_are_serializable()`

### `tests.unit.core`
- Arquivo: `tests/unit/core/__init__.py`

### `tests.unit.core.test_accumulator`
- Arquivo: `tests/unit/core/test_accumulator.py`
- Imports:
  - `mod:collections`
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestStatisticsAccumulator` (bases: (sem base explícita))

### `tests.unit.core.test_config`
- Arquivo: `tests/unit/core/test_config.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestNeuronConfig` (bases: (sem base explícita))
- Funções:
  - `test_hebbian_ltd_ratio_must_be_non_negative()`

### `tests.unit.core.test_config_advanced_order_flags`
- Arquivo: `tests/unit/core/test_config_advanced_order_flags.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
- Funções:
  - `test_accepts_stdp_source_and_ltd_rule_flags()`
  - `test_rejects_invalid_order_flags(kwargs)`

### `tests.unit.core.test_dendrite`
- Arquivo: `tests/unit/core/test_dendrite.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestMPJRDDendrite` (bases: (sem base explícita))

### `tests.unit.core.test_factory`
- Arquivo: `tests/unit/core/test_factory.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.core`
  - `mod:pyfolds.core.factory`
  - `mod:pyfolds.wave`
- Funções:
  - `test_create_neuron_infers_standard_type()`
  - `test_create_neuron_infers_wave_type()`
  - `test_factory_raises_for_unregistered_type()`

### `tests.unit.core.test_health_monitor`
- Arquivo: `tests/unit/core/test_health_monitor.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.monitoring`
  - `mod:torch`
- Funções:
  - `test_health_monitor_runs_and_scores()`
  - `test_weight_integrity_monitor_detects_change()`
  - `test_weight_integrity_monitor_respects_interval()`

### `tests.unit.core.test_homeostasis`
- Arquivo: `tests/unit/core/test_homeostasis.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestHomeostasisController` (bases: (sem base explícita))

### `tests.unit.core.test_input_validation_contract`
- Arquivo: `tests/unit/core/test_input_validation_contract.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.core.neuron`
  - `mod:pytest`
  - `mod:torch`
- Funções:
  - `_cfg()`
  - `test_layer_prepare_input_accepts_supported_shapes()`
  - `test_layer_prepare_input_rejects_invalid_shape()`
  - `test_neuron_rejects_input_from_different_device()`
  - `test_neuron_rejects_non_tensor_input()`

### `tests.unit.core.test_monitoring_and_checkpoint`
- Arquivo: `tests/unit/core/test_monitoring_and_checkpoint.py`
- Imports:
  - `mod:datetime`
  - `mod:json`
  - `mod:pathlib`
  - `mod:pyfolds`
  - `mod:pyfolds.monitoring`
  - `mod:pyfolds.serialization`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `DummyNeuron` (bases: (sem base explícita))
- Funções:
  - `test_health_check_critical_for_dead_neurons()`
  - `test_health_check_uses_fallback_metrics_from_get_metrics_contract()`
  - `test_model_integrity_monitor_detects_unexpected_mutation()`
  - `test_model_integrity_monitor_initializes_hash_on_first_check()`
  - `test_versioned_checkpoint_load_secure_fails_on_hash_mismatch(tmp_path)`
  - `test_versioned_checkpoint_load_secure_validates_hash_and_shapes(tmp_path)`
  - `test_versioned_checkpoint_metadata_created_at_is_utc(tmp_path)`
  - `test_versioned_checkpoint_safetensors_roundtrip(tmp_path)`
  - `test_versioned_checkpoint_save_and_load(tmp_path)`
  - `test_versioned_checkpoint_shape_validation_raises_on_mismatch(tmp_path)`
  - `test_weight_integrity_monitor_detects_mutation_between_checks()`

### `tests.unit.core.test_neuromodulation`
- Arquivo: `tests/unit/core/test_neuromodulation.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestNeuromodulator` (bases: (sem base explícita))

### `tests.unit.core.test_neuron`
- Arquivo: `tests/unit/core/test_neuron.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.utils.types`
  - `mod:pytest`
  - `mod:torch`
- Classes:
  - `TestMPJRDNeuron` (bases: (sem base explícita))

### `tests.unit.core.test_neuron_v2`
- Arquivo: `tests/unit/core/test_neuron_v2.py`
- Imports:
  - `mod:concurrent.futures`
  - `mod:pyfolds`
  - `mod:torch`
- Funções:
  - `test_cooperative_sum_uses_multiple_dendrites()`
  - `test_forward_shapes_v2(small_config, batch_size)`
  - `test_step_id_thread_safe_increment_v2(small_config)`
  - `test_theta_eff_caps_unreachable_threshold()`
  - `test_vectorization_and_forward_integrity_batch64()`

### `tests.unit.core.test_synapse`
- Arquivo: `tests/unit/core/test_synapse.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.utils.types`
  - `mod:pytest`
  - `mod:torch`
  - `mod:unittest`
- Classes:
  - `TestMPJRDSynapse` (bases: (sem base explícita))

### `tests.unit.network.test_network_edge_cases`
- Arquivo: `tests/unit/network/test_network_edge_cases.py`
- Imports:
  - `mod:pyfolds`
  - `mod:torch`
- Funções:
  - `test_prepare_input_with_single_dendrite_avoids_division_by_zero()`

### `tests.unit.neuron.test_adaptation_sfa`
- Arquivo: `tests/unit/neuron/test_adaptation_sfa.py`
- Imports:
  - `mod:pyfolds`
  - `mod:torch`
- Funções:
  - `test_sfa_applies_before_threshold_and_reduces_spiking_probability()`

### `tests.unit.neuron.test_backprop_bap`
- Arquivo: `tests/unit/neuron/test_backprop_bap.py`
- Imports:
  - `mod:pyfolds`
  - `mod:torch`
- Funções:
  - `test_bap_amplification_changes_dendritic_computation_and_clamps_gain()`

### `tests.unit.neuron.test_contract_conformance`
- Arquivo: `tests/unit/neuron/test_contract_conformance.py`
- Imports:
  - `mod:numpy`
  - `mod:pyfolds.contracts`
  - `mod:pytest`
  - `mod:torch`
- Funções:
  - `test_torch_and_tf_contract_conformance_with_same_artificial_input_and_tolerance()`
  - `test_torch_contract_invariants_order_and_time_step_end_of_step()`

### `tests.unit.neuron.test_integration_stability`
- Arquivo: `tests/unit/neuron/test_integration_stability.py`
- Imports:
  - `mod:pyfolds`
  - `mod:torch`
- Funções:
  - `test_forward_clamps_non_finite_dendritic_sum(monkeypatch)`

### `tests.unit.neuron.test_refractory`
- Arquivo: `tests/unit/neuron/test_refractory.py`
- Imports:
  - `mod:pyfolds`
  - `mod:torch`
- Funções:
  - `test_absolute_refractory_is_inviolable()`

### `tests.unit.neuron.test_stp_stdp_contracts`
- Arquivo: `tests/unit/neuron/test_stp_stdp_contracts.py`
- Imports:
  - `mod:pyfolds`
  - `mod:torch`
- Funções:
  - `test_stdp_reads_pre_stp_input_for_pre_spikes_and_ltd_uses_pre_spike_gate()`

### `tests.unit.neuron.test_time_counter`
- Arquivo: `tests/unit/neuron/test_time_counter.py`
- Imports:
  - `mod:pyfolds`
  - `mod:torch`
- Funções:
  - `test_time_counter_increments_at_end_of_forward_step()`

### `tests.unit.serialization.test_foldio`
- Arquivo: `tests/unit/serialization/test_foldio.py`
- Imports:
  - `mod:mmap`
  - `mod:numpy`
  - `mod:pyfolds`
  - `mod:pyfolds.core.neuron`
  - `mod:pyfolds.serialization`
  - `mod:pyfolds.serialization.foldio`
  - `mod:pytest`
  - `mod:struct`
  - `mod:torch`
  - `mod:unittest`
- Funções:
  - `_build_neuron(enable_telemetry)`
  - `_build_writer_with_chunk(tmp_path)`
  - `_failing_write_after_first_call(original_write)`
  - `test_crc32c_matches_known_vector()`
  - `test_detects_corruption(tmp_path)`
  - `test_ecc_from_protection_mapping()`
  - `test_ecc_roundtrip_if_available(tmp_path)`
  - `test_fold_manifest_includes_governance_sections(tmp_path)`
  - `test_fold_reader_bounds_validation_negative_values(tmp_path)`
  - `test_fold_reader_bounds_validation_with_mmap(tmp_path)`
  - `test_fold_reader_exit_closes_file_even_if_mmap_close_fails(tmp_path)`
  - `test_fold_reader_header_len_validation(tmp_path)`
  - `test_fold_reader_index_offset_validation(tmp_path)`
  - `test_fold_reader_index_size_validation(tmp_path)`
  - `test_fold_reader_reports_magic_values(tmp_path)`
  - `test_fold_roundtrip_and_peek(tmp_path)`
  - `test_fold_roundtrip_preserves_state_dict_after_forward_steps(tmp_path)`
  - `test_fold_signature_roundtrip_if_cryptography_available(tmp_path)`
  - `test_fold_writer_finalize_wraps_failures_with_phase(tmp_path, monkeypatch, phase, patcher, error_message)`
  - `test_fold_writer_finalize_wraps_io_failure_with_phase_context(tmp_path, monkeypatch)`
  - `test_hierarchical_hashes_present_in_metadata(tmp_path)`
  - `test_training_then_save_with_telemetry_and_history_and_nuclear_arrays(tmp_path)`
  - `test_validate_safetensors_payload_rejects_invalid_json()`
  - `test_validate_safetensors_payload_rejects_oversized_header()`

### `tests.unit.telemetry`
- Arquivo: `tests/unit/telemetry/__init__.py`

### `tests.unit.telemetry.test_controller`
- Arquivo: `tests/unit/telemetry/test_controller.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
- Classes:
  - `TestTelemetryController` (bases: (sem base explícita))

### `tests.unit.telemetry.test_decorator`
- Arquivo: `tests/unit/telemetry/test_decorator.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
- Classes:
  - `TestTelemetryDecorator` (bases: (sem base explícita))

### `tests.unit.telemetry.test_events`
- Arquivo: `tests/unit/telemetry/test_events.py`
- Imports:
  - `mod:dataclasses`
  - `mod:time`
  - `mod:typing`
- Classes:
  - `TelemetryEvent` (bases: (sem base explícita))
- Funções:
  - `commit_event(step_id, mode, neuron_id)`
  - `commit_event_lazy(step_id, mode, payload_fn, neuron_id)`
  - `forward_event(step_id, mode, neuron_id)`
  - `forward_event_lazy(step_id, mode, payload_fn, neuron_id)`
  - `sleep_event(step_id, mode, neuron_id)`
  - `sleep_event_lazy(step_id, mode, payload_fn, neuron_id)`

### `tests.unit.telemetry.test_ringbuffer`
- Arquivo: `tests/unit/telemetry/test_ringbuffer.py`
- Imports:
  - `mod:pyfolds.telemetry`
  - `mod:pytest`
  - `mod:threading`
- Classes:
  - `TestRingBuffer` (bases: (sem base explícita))

### `tests.unit.telemetry.test_sinks`
- Arquivo: `tests/unit/telemetry/test_sinks.py`
- Imports:
  - `mod:json`
  - `mod:pyfolds`
  - `mod:pytest`
- Classes:
  - `TestBufferedJSONLinesSink` (bases: (sem base explícita))
  - `TestConsoleSink` (bases: (sem base explícita))
  - `TestDistributorSink` (bases: (sem base explícita))
  - `TestJSONLinesSink` (bases: (sem base explícita))
  - `TestMemorySink` (bases: (sem base explícita))

### `tests.unit.telemetry.test_types`
- Arquivo: `tests/unit/telemetry/test_types.py`
- Imports:
  - `mod:pyfolds.telemetry`
- Classes:
  - `TestPayloadTypes` (bases: (sem base explícita))

### `tests.unit.test_backend_contracts`
- Arquivo: `tests/unit/test_backend_contracts.py`
- Imports:
  - `mod:importlib.util`
  - `mod:numpy`
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Funções:
  - `_tf_forward_sequence_equivalent(x_seq)`
  - `test_import_and_object_construction_v1_v2()`
  - `test_tf_backend_conditional_shape_state_contracts()`
  - `test_torch_backend_shape_and_state_contracts()`
  - `test_torch_backend_v2_accepts_multidim_batch_contract()`

### `tests.unit.test_design_improvements`
- Arquivo: `tests/unit/test_design_improvements.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.advanced`
  - `mod:pyfolds.core.base`
  - `mod:pyfolds.factory`
  - `mod:pyfolds.network`
  - `mod:pyfolds.utils`
  - `mod:pyfolds.utils.context`
  - `mod:pytest`
- Classes:
  - `DummyNeuron` (bases: BaseNeuron)
- Funções:
  - `test_factory_creates_builtin_types()`
  - `test_factory_custom_registry_and_unknown_type()`
  - `test_learning_mode_context_restores_even_on_error()`
  - `test_network_builder_connects_layers_and_builds()`

### `tests.unit.test_layer_neuron_class`
- Arquivo: `tests/unit/test_layer_neuron_class.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.core`
  - `mod:pytest`
  - `mod:torch`
- Funções:
  - `test_layer_accepts_neuron_v2()`
  - `test_layer_forwards_dt_to_neurons(monkeypatch)`
  - `test_layer_has_no_legacy_neuron_class_attr()`
  - `test_layer_rejects_invalid_neuron_cls()`

### `tests.unit.test_learning_mode_consistency`
- Arquivo: `tests/unit/test_learning_mode_consistency.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.utils.types`
  - `mod:pytest`
  - `mod:torch`
- Funções:
  - `_make_layer()`
  - `test_layer_accepts_mode_enum()`
  - `test_layer_accepts_mode_string()`
  - `test_layer_rejects_invalid_mode()`
  - `test_learning_mode_from_string()`
  - `test_learning_mode_single_source()`
  - `test_network_accepts_mode_string()`
  - `test_network_rejects_invalid_mode()`

### `tests.unit.test_public_import_surface`
- Arquivo: `tests/unit/test_public_import_surface.py`
- Imports:
  - `mod:packaging.version`
  - `mod:pyfolds`
  - `mod:warnings`
- Funções:
  - `test_public_all_exports_are_importable()`
  - `test_telemetry_controller_basic_flow()`
  - `test_v1_aliases_emit_deprecation_warning_and_match_v2_targets_until_2_0()`
  - `test_v2_surface_is_canonical_and_instantiable()`

### `tests.unit.test_run_pyfolds_runner`
- Arquivo: `tests/unit/test_run_pyfolds_runner.py`
- Imports:
  - `mod:pathlib`
  - `mod:subprocess`
  - `mod:sys`
- Funções:
  - `_run(script_path)`
  - `test_runner_logs_runtime_error_and_propagates_exit_code(tmp_path)`
  - `test_runner_logs_syntax_error(tmp_path)`
  - `test_runner_retry_and_metrics_csv(tmp_path)`
  - `test_runner_tee_prints_progress(tmp_path)`
  - `test_runner_timeout_watchdog(tmp_path)`

### `tests.unit.test_tf_backend`
- Arquivo: `tests/unit/test_tf_backend.py`
- Imports:
  - `mod:importlib`
  - `mod:importlib.util`
  - `mod:pytest`
  - `mod:sys`
- Funções:
  - `test_importing_pyfolds_still_works_without_tensorflow()`
  - `test_tf_backend_guard_when_tensorflow_is_missing(monkeypatch)`
  - `test_tf_cell_state_and_step_contract()`
  - `test_tf_layer_integrates_with_keras_rnn()`

### `tests.unit.utils`
- Arquivo: `tests/unit/utils/__init__.py`

### `tests.unit.utils.test_device`
- Arquivo: `tests/unit/utils/test_device.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.utils`
  - `mod:torch`
- Classes:
  - `TestDevice` (bases: (sem base explícita))

### `tests.unit.utils.test_logging`
- Arquivo: `tests/unit/utils/test_logging.py`
- Imports:
  - `mod:logging`
  - `mod:pathlib`
  - `mod:pyfolds.utils.logging`
  - `mod:time`
- Funções:
  - `_build_record(level, message)`
  - `test_circular_buffer_flushes_immediately_on_error(tmp_path)`
  - `test_circular_buffer_lazy_flush_by_interval(tmp_path)`

### `tests.unit.utils.test_math`
- Arquivo: `tests/unit/utils/test_math.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.utils`
  - `mod:pyfolds.utils.math`
  - `mod:torch`
- Classes:
  - `TestMath` (bases: (sem base explícita))

### `tests.unit.utils.test_types`
- Arquivo: `tests/unit/utils/test_types.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pyfolds.utils.types`
- Classes:
  - `TestTypes` (bases: (sem base explícita))

### `tests.unit.utils.test_utils`
- Arquivo: `tests/unit/utils/test_utils.py`
- Imports:
  - `mod:logging`
  - `mod:os`
  - `mod:pathlib`
  - `mod:pyfolds.utils`
  - `mod:pyfolds.utils.logging`
  - `mod:pytest`
  - `mod:re`
  - `mod:subprocess`
  - `mod:tempfile`
  - `mod:torch`
- Classes:
  - `TestDevice` (bases: (sem base explícita))
  - `TestIntegration` (bases: (sem base explícita))
  - `TestLogging` (bases: (sem base explícita))
  - `TestMath` (bases: (sem base explícita))
  - `TestTypes` (bases: (sem base explícita))

### `tests.unit.utils.test_validation`
- Arquivo: `tests/unit/utils/test_validation.py`
- Imports:
  - `mod:pyfolds`
  - `mod:pytest`
  - `mod:torch`
- Funções:
  - `test_neuron_forward_validation_dtype(small_config)`
  - `test_neuron_forward_validation_ndim(small_config)`

### `tests.unit.wave.test_wave_config`
- Arquivo: `tests/unit/wave/test_wave_config.py`
- Imports:
  - `mod:pyfolds.wave`
  - `mod:pytest`
- Funções:
  - `test_wave_config_defaults_valid()`
  - `test_wave_config_rejects_invalid_buffer()`

### `tests.unit.wave.test_wave_layer_network`
- Arquivo: `tests/unit/wave/test_wave_layer_network.py`
- Imports:
  - `mod:pyfolds`
  - `mod:torch`
- Funções:
  - `test_wave_layer_exposes_wave_outputs()`
  - `test_wave_network_forwards_layer_kwargs()`

### `tests.unit.wave.test_wave_neuron`
- Arquivo: `tests/unit/wave/test_wave_neuron.py`
- Imports:
  - `mod:pyfolds.wave`
  - `mod:torch`
- Funções:
  - `test_cooperative_integration_uses_multiple_dendrites()`
  - `test_wave_outputs_quadrature_and_phase_range()`
  - `test_wave_step_id_thread_safe_increment()`
