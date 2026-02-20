"""Telemetry system for MPJRD neurons.

This module provides:
- Structured events (forward, commit, sleep)
- Thread-safe circular buffer
- Multiple sinks (memory, console, JSON, distributor)
- Controller with off/light/heavy profiles
- Decorator for automatic telemetry
- Typed payloads

Basic usage:
    from pyfolds.telemetry import TelemetryConfig, TelemetryController, forward_event
    
    cfg = TelemetryConfig(profile="light", sample_every=50)
    telem = TelemetryController(cfg)
    
    if telem.enabled() and telem.should_emit(step_id):
        telem.emit(forward_event(step_id, mode="online", spike_rate=0.15))
"""

from .events import (
    TelemetryEvent,
    forward_event,
    forward_event_lazy,
    commit_event,
    commit_event_lazy,
    sleep_event,
    sleep_event_lazy,
)
from .ringbuffer import RingBuffer
from .sinks import (
    Sink,
    NoOpSink,
    MemorySink,
    ConsoleSink,
    JSONLinesSink,
    BufferedJSONLinesSink,
    DistributorSink,
)
from .controller import TelemetryController, TelemetryConfig, Profile, TelemetryProfile
from .decorator import telemetry
from .types import ForwardPayload, CommitPayload, SleepPayload

__version__ = "2.1.1"

__all__ = [
    # Events
    "TelemetryEvent",
    "forward_event",
    "commit_event",
    "sleep_event",
    "forward_event_lazy",
    "commit_event_lazy",
    "sleep_event_lazy",
    
    # Buffer
    "RingBuffer",
    
    # Sinks
    "Sink",
    "NoOpSink",
    "MemorySink",
    "ConsoleSink",
    "JSONLinesSink",
    "BufferedJSONLinesSink",
    "DistributorSink",
    
    # Controller
    "TelemetryController",
    "TelemetryConfig",
    "Profile",
    "TelemetryProfile",
    
    # Decorator
    "telemetry",
    
    # Types
    "ForwardPayload",
    "CommitPayload",
    "SleepPayload",
    
    # Metadata
    "__version__",
]