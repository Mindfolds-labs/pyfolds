"""PyFolds - Core Neural Computation Framework

Módulos disponíveis diretamente:
    - MPJRDConfig, MPJRDNeuron: Core neural computation
    - MPJRDLayer, MPJRDNetwork: Network layers
    - TelemetryController, MemorySink: Monitoring (for MindMetrics/MindAudit)
    - LearningMode, ConnectionType: Enums and types

Uso básico:
    from pyfolds import MPJRDConfig, MPJRDNeuron, TelemetryController
    
    cfg = MPJRDConfig(n_dendrites=4)
    neuron = MPJRDNeuron(cfg)
    
    # Telemetria para MindMetrics
    telem = TelemetryController()
"""

__version__ = "2.0.0"

# ===== CORE COMPONENTS =====
from .core.config import MPJRDConfig
from .core.neuron import MPJRDNeuron
from .layers import MPJRDLayer, MPJRDWaveLayer
from .network import MPJRDNetwork, MPJRDWaveNetwork
from .core.neuron_v2 import MPJRDNeuronV2
from .utils.types import LearningMode, ConnectionType
from .wave import MPJRDWaveConfig, MPJRDWaveNeuron

# ===== TELEMETRY (para MindMetrics/MindAudit) =====
from .telemetry import (
    # Controller
    TelemetryController,
    TelemetryConfig,
    
    # Sinks
    Sink,
    NoOpSink,
    MemorySink,
    ConsoleSink,
    JSONLinesSink,
    DistributorSink,
    
    # Events
    forward_event,
    commit_event,
    sleep_event,
    forward_event_lazy,
    commit_event_lazy,
    sleep_event_lazy,
    
    # Buffer
    RingBuffer,
    
    # Decorator
    telemetry,
    
    # Types
    ForwardPayload,
    CommitPayload,
    SleepPayload,
)

# ===== ADVANCED (optional) =====
try:
    from .advanced import (
        MPJRDNeuronAdvanced,
        MPJRDLayerAdvanced,
        MPJRDWaveNeuronAdvanced,
        MPJRDWaveLayerAdvanced,
    )
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    MPJRDNeuronAdvanced = None
    MPJRDLayerAdvanced = None
    MPJRDWaveNeuronAdvanced = None
    MPJRDWaveLayerAdvanced = None

# ===== EXPORTS =====
__all__ = [
    # Core
    "MPJRDConfig",
    "MPJRDNeuron",
    "MPJRDNeuronV2",
    "MPJRDLayer",
    "MPJRDNetwork",
    "MPJRDWaveLayer",
    "MPJRDWaveNetwork",
    "MPJRDWaveConfig",
    "MPJRDWaveNeuron",
    
    # Types
    "LearningMode",
    "ConnectionType",
    
    # Telemetry (para MindMetrics/MindAudit)
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
]

# Add advanced if available
if ADVANCED_AVAILABLE:
    __all__.extend([
        "MPJRDNeuronAdvanced",
        "MPJRDLayerAdvanced",
        "MPJRDWaveNeuronAdvanced",
        "MPJRDWaveLayerAdvanced",
    ])
