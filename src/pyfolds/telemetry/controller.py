"""Telemetry controller for MPJRD neurons."""

import random
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Optional, Literal, List, Dict, Any, TypedDict
from .events import TelemetryEvent
from .sinks import Sink, NoOpSink, MemorySink
from .ringbuffer import RingBuffer

logger = logging.getLogger(__name__)
Profile = Literal["off", "light", "heavy"]


class TelemetryStats(TypedDict):
    """Typed statistics for telemetry system."""
    profile: Profile
    sample_every: int
    step_count: int
    sink_type: str
    buffer_size: int
    enabled: bool
    events_collected: int
    estimated_memory_mb: float
    sampling_rate: str


@dataclass
class TelemetryConfig:
    """Configuration for telemetry system."""
    
    profile: Profile = "off"
    sample_every: int = 1
    memory_capacity: int = 512
    
    def __post_init__(self) -> None:
        """Validate configuration eagerly."""
        # Validate profile
        if self.profile not in ("off", "light", "heavy"):
            raise ValueError(f"Invalid profile: {self.profile}. "
                           f"Must be 'off', 'light', or 'heavy'")
        
        # Validate sample_every
        if self.sample_every < 1:
            raise ValueError(f"sample_every must be >= 1, got {self.sample_every}")
        
        # Validate memory_capacity
        if self.memory_capacity < 1:
            raise ValueError(f"memory_capacity must be >= 1, got {self.memory_capacity}")
        
        # Apply profile-specific defaults
        if self.profile == "light" and self.sample_every == 1:
            self.sample_every = 50  # Light profile default
        elif self.profile == "heavy" and self.sample_every != 1:
            self.sample_every = 1   # Heavy = every step


class TelemetryController:
    """
    Telemetry controller with thread-safe event collection.
    
    This controller manages event filtering, sampling, and routing to sinks.
    All public methods are thread-safe.
    """
    
    def __init__(self, cfg: Optional[TelemetryConfig] = None, sink: Optional[Sink] = None):
        """
        Initialize telemetry controller.
        
        Args:
            cfg: Telemetry configuration (uses defaults if None)
            sink: Destination sink (auto-selected based on profile if None)
        """
        self.cfg = cfg or TelemetryConfig()
        
        # Thread safety
        self._lock = Lock()
        self._step_count = 0
        
        # Select sink based on profile
        if sink is not None:
            self.sink = sink
        elif self.cfg.profile == "off":
            self.sink = NoOpSink()
        else:
            self.sink = MemorySink(self.cfg.memory_capacity)
    
    @property
    def step_count(self) -> int:
        """Thread-safe step counter."""
        with self._lock:
            return self._step_count
    
    def _increment_step(self) -> None:
        """Thread-safe step increment."""
        with self._lock:
            self._step_count += 1
    
    def enabled(self) -> bool:
        """Is telemetry enabled?"""
        return self.cfg.profile != "off"
    
    def should_emit(self, step_id: int) -> bool:
        """
        Should emit event at this step based on sample_every?
        
        Args:
            step_id: Current step ID
            
        Returns:
            True if should emit
        """
        if self.cfg.profile == "off":
            return False
        
        if self.cfg.profile == "heavy":
            return True
        
        # Light profile: emit every sample_every steps
        every = max(1, self.cfg.sample_every)
        return (step_id % every) == 0
    
    def should_emit_sample(self, sample_rate: float) -> bool:
        """
        Probabilistic sampling independent of step_id.
        
        Args:
            sample_rate: Probability of emitting (0.0 to 1.0)
            
        Returns:
            True if should emit based on random sampling
        """
        if self.cfg.profile == "off":
            return False
        if sample_rate >= 1.0:
            return True
        return random.random() < sample_rate
    
    def emit(self, event: TelemetryEvent) -> None:
        """
        Emit a telemetry event (thread-safe).
        
        Args:
            event: Event to emit (may have lazy payload)
        """
        if not self.should_emit(event.step_id):
            return
        
        try:
            self.sink.emit(event)
        except Exception as e:
            logger.error(f"Failed to emit telemetry event: {e}")
        
        self._increment_step()
    
    def snapshot(self) -> List[Dict[str, Any]]:
        """
        Return snapshot of stored events (thread-safe).
        
        Returns:
            List of event payloads
        """
        if isinstance(self.sink, MemorySink):
            with self._lock:
                return [e.payload for e in self.sink.buffer.snapshot()]
        return []
    
    def clear(self) -> None:
        """Clear memory buffer (thread-safe)."""
        if isinstance(self.sink, MemorySink):
            with self._lock:
                self.sink.buffer = RingBuffer[TelemetryEvent](self.cfg.memory_capacity)
    
    def get_stats(self) -> TelemetryStats:
        """Return telemetry system statistics."""
        buffer_size = 0
        if isinstance(self.sink, MemorySink):
            buffer_size = len(self.sink.buffer)
        
        # Estimate memory (rough)
        estimated_memory_mb = (buffer_size * 1024) / (1024 * 1024)
        
        return {
            'profile': self.cfg.profile,
            'sample_every': self.cfg.sample_every,
            'step_count': self.step_count,
            'sink_type': type(self.sink).__name__,
            'buffer_size': buffer_size,
            'enabled': self.enabled(),
            'events_collected': buffer_size,
            'estimated_memory_mb': estimated_memory_mb,
            'sampling_rate': f"1/{self.cfg.sample_every}",
        }
    
    def get_telemetry_metrics(self) -> Dict[str, Any]:
        """Return metrics about telemetry overhead."""
        return self.get_stats()