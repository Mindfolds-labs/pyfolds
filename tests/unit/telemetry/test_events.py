"""Telemetry events for MPJRD neurons."""

import time
from dataclasses import dataclass, field
from typing import Literal, Any, Dict, Optional, Union, Callable

Phase = Literal["forward", "commit", "sleep", "mode_change"]

# Type for payload that can be eager or lazy
PayloadType = Union[Dict[str, Any], Callable[[], Dict[str, Any]]]


@dataclass(frozen=True)
class TelemetryEvent:
    """
    Base telemetry event with lazy evaluation support.
    
    Attributes:
        step_id: Current step ID
        phase: Event phase (forward, commit, sleep)
        mode: Learning mode
        _payload: Dict or function returning payload
        timestamp: High-precision timestamp (perf_counter)
        wall_time: Wall time for temporal analysis
        neuron_id: Optional neuron ID
    """
    step_id: int
    phase: Phase
    mode: str
    _payload: PayloadType
    timestamp: float = field(default_factory=lambda: time.perf_counter())  # ⚡ High precision
    wall_time: float = field(default_factory=lambda: time.time())          # ⏰ For reference
    neuron_id: Optional[str] = None
    
    @property
    def payload(self) -> Dict[str, Any]:
        """Evaluate lazy payload if needed."""
        if callable(self._payload):
            return self._payload()
        return self._payload


def forward_event(step_id: int, mode: str, neuron_id: Optional[str] = None, 
                  **payload: Any) -> TelemetryEvent:
    """
    Create eager forward event.
    
    Args:
        step_id: Step ID
        mode: Learning mode
        neuron_id: Optional neuron ID
        **payload: Metrics to collect
    """
    return TelemetryEvent(
        step_id=step_id,
        phase="forward",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload
    )


# ✅ CORRIGIDO: payload_fn movido ANTES de neuron_id
def forward_event_lazy(step_id: int, mode: str,
                       payload_fn: Callable[[], Dict[str, Any]],
                       neuron_id: Optional[str] = None) -> TelemetryEvent:
    """
    Create lazy forward event (CPU-efficient).
    
    Args:
        step_id: Step ID
        mode: Learning mode
        payload_fn: Function returning payload (only called if event is emitted)
        neuron_id: Optional neuron ID
    
    Example:
        >>> telem.emit(forward_event_lazy(
        ...     step_id=step,
        ...     mode="online",
        ...     payload_fn=lambda: {
        ...         'spike_rate': neuron.get_spike_rate(),  # Only calculated if needed
        ...         'theta': neuron.theta.item()
        ...     }
        ... ))
    """
    return TelemetryEvent(
        step_id=step_id,
        phase="forward",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload_fn
    )


def commit_event(step_id: int, mode: str, neuron_id: Optional[str] = None, 
                 **payload: Any) -> TelemetryEvent:
    """Create eager commit event."""
    return TelemetryEvent(
        step_id=step_id,
        phase="commit",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload
    )


# ✅ CORRIGIDO: payload_fn movido ANTES de neuron_id
def commit_event_lazy(step_id: int, mode: str,
                      payload_fn: Callable[[], Dict[str, Any]],
                      neuron_id: Optional[str] = None) -> TelemetryEvent:
    """Create lazy commit event."""
    return TelemetryEvent(
        step_id=step_id,
        phase="commit",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload_fn
    )


def sleep_event(step_id: int, mode: str, neuron_id: Optional[str] = None, 
                **payload: Any) -> TelemetryEvent:
    """Create eager sleep event."""
    return TelemetryEvent(
        step_id=step_id,
        phase="sleep",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload
    )


# ✅ CORRIGIDO: payload_fn movido ANTES de neuron_id
def sleep_event_lazy(step_id: int, mode: str,
                     payload_fn: Callable[[], Dict[str, Any]],
                     neuron_id: Optional[str] = None) -> TelemetryEvent:
    """Create lazy sleep event."""
    return TelemetryEvent(
        step_id=step_id,
        phase="sleep",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload_fn
    )