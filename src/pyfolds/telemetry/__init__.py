"""Sistema de telemetria para neurônios MPJRD

Este módulo fornece:
- Eventos estruturados (forward, commit, sleep)
- Buffer circular thread-safe
- Múltiplos sinks (memória, console, JSON, distribuidor)
- Controlador com perfis off/light/heavy
- Decorador para telemetria automática
- Payloads tipados

Uso básico:
    from pyfolds.telemetry import TelemetryConfig, TelemetryController, forward_event
    
    cfg = TelemetryConfig(profile="light", sample_every=50)
    telem = TelemetryController(cfg)
    
    if telem.enabled() and telem.should_emit(step_id):
        telem.emit(forward_event(step_id, mode="online", spike_rate=0.15))
"""

from .events import TelemetryEvent, forward_event, commit_event, sleep_event
from .ringbuffer import RingBuffer
from .sinks import Sink, NoOpSink, MemorySink, ConsoleSink, JSONLinesSink, DistributorSink
from .controller import TelemetryController, TelemetryConfig, Profile
from .decorator import telemetry
from .types import ForwardPayload, CommitPayload, SleepPayload

__version__ = "1.2.0"  # ⬆️ Versão atualizada!

__all__ = [
    # Eventos
    "TelemetryEvent",
    "forward_event",
    "commit_event",
    "sleep_event",
    
    # Buffer
    "RingBuffer",
    
    # Sinks
    "Sink",
    "NoOpSink",
    "MemorySink",
    "ConsoleSink",
    "JSONLinesSink",
    "DistributorSink",  # ✅ NOVO!
    
    # Controlador
    "TelemetryController",
    "TelemetryConfig",
    "Profile",
    
    # Decorador
    "telemetry",
    
    # Tipos
    "ForwardPayload",
    "CommitPayload",
    "SleepPayload",
    
    # Metadados
    "__version__",
]