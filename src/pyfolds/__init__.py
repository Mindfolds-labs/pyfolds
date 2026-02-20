"""PyFolds public package surface.

This module exposes the v2 public API names as canonical imports and keeps
v1 names as compatibility aliases with deprecation warnings.
"""

from typing import Any, Dict
import warnings

__version__ = "2.0.1"

# ===== CORE V2 SURFACE =====
from .core.config import MPJRDConfig as NeuronConfig
from .core.base import BaseNeuron, BasePlasticityRule
from .core.neuron import MPJRDNeuron
from .layers import MPJRDLayer as AdaptiveNeuronLayer, MPJRDWaveLayer
from .network import MPJRDNetwork as SpikingNetwork, MPJRDWaveNetwork, NetworkBuilder
from .core.neuron_v2 import MPJRDNeuronV2
from .utils.types import LearningMode, ConnectionType
from .utils.context import learning_mode
from .wave import MPJRDWaveConfig, MPJRDWaveNeuron
from .core.factory import NeuronFactory, NeuronType
from .serialization import (
    VersionedCheckpoint,
    FoldReader,
    FoldWriter,
    FoldSecurityError,
    save_fold_or_mind,
    load_fold_or_mind,
    peek_fold_or_mind,
    peek_mind,
    read_nuclear_arrays,
    is_mind,
    NoECC,
    ReedSolomonECC,
    ecc_from_protection,
)
from .monitoring import (
    HealthStatus,
    NeuronHealthCheck,
    MindControl,
    MindControlSink,
    MutationCommand,
)

# High-level state contract returned by forward-like calls.
NeuronState = Dict[str, Any]

# ===== TELEMETRY =====
from .telemetry import (
    TelemetryController,
    TelemetryConfig,
    Sink,
    NoOpSink,
    MemorySink,
    ConsoleSink,
    JSONLinesSink,
    DistributorSink,
    forward_event,
    commit_event,
    sleep_event,
    forward_event_lazy,
    commit_event_lazy,
    sleep_event_lazy,
    RingBuffer,
    telemetry,
    ForwardPayload,
    CommitPayload,
    SleepPayload,
)

from .advanced import (
    MPJRDNeuronAdvanced,
    MPJRDLayerAdvanced,
    MPJRDWaveNeuronAdvanced,
    MPJRDWaveLayerAdvanced,
)
ADVANCED_AVAILABLE = True

_V1_ALIAS_MAP = {
    "MPJRDConfig": "NeuronConfig",
    "MPJRDLayer": "AdaptiveNeuronLayer",
    "MPJRDNetwork": "SpikingNetwork",
}

__all__ = [
    # v2 canonical names
    "NeuronConfig",
    "NeuronState",
    "MPJRDNeuron",
    "MPJRDNeuronV2",
    "AdaptiveNeuronLayer",
    "SpikingNetwork",
    "MPJRDWaveLayer",
    "MPJRDWaveNetwork",
    "NetworkBuilder",
    # Core supporting exports
    "BaseNeuron",
    "BasePlasticityRule",
    "MPJRDWaveConfig",
    "MPJRDWaveNeuron",
    "NeuronFactory",
    "NeuronType",
    "VersionedCheckpoint",
    "FoldReader",
    "FoldWriter",
    "FoldSecurityError",
    "save_fold_or_mind",
    "load_fold_or_mind",
    "peek_fold_or_mind",
    "peek_mind",
    "read_nuclear_arrays",
    "is_mind",
    "NoECC",
    "ReedSolomonECC",
    "ecc_from_protection",
    "HealthStatus",
    "NeuronHealthCheck",
    "MindControl",
    "MindControlSink",
    "MutationCommand",
    # Types/helpers
    "LearningMode",
    "ConnectionType",
    "learning_mode",
    # Telemetry
    "TelemetryController",
    "TelemetryConfig",
    "Sink",
    "NoOpSink",
    "MemorySink",
    "ConsoleSink",
    "JSONLinesSink",
    "DistributorSink",
    "forward_event",
    "commit_event",
    "sleep_event",
    "forward_event_lazy",
    "commit_event_lazy",
    "sleep_event_lazy",
    "RingBuffer",
    "telemetry",
    "ForwardPayload",
    "CommitPayload",
    "SleepPayload",
    # Advanced
    "MPJRDNeuronAdvanced",
    "MPJRDLayerAdvanced",
    "MPJRDWaveNeuronAdvanced",
    "MPJRDWaveLayerAdvanced",
    # v1 compatibility aliases
    "MPJRDConfig",
    "MPJRDLayer",
    "MPJRDNetwork",
]


def __getattr__(name: str):
    """Provide deprecated v1 aliases while keeping v2 names canonical."""
    if name in _V1_ALIAS_MAP:
        target_name = _V1_ALIAS_MAP[name]
        warnings.warn(
            f"'pyfolds.{name}' está depreciado e será removido em versão futura; "
            f"use 'pyfolds.{target_name}'",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[target_name]
    raise AttributeError(f"module 'pyfolds' has no attribute '{name}'")
