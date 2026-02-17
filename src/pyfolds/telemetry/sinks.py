"""Sinks (destinations) for telemetry events."""

import json
import logging
from abc import ABC, abstractmethod
from typing import List, Union
from pathlib import Path
from .events import TelemetryEvent
from .ringbuffer import RingBuffer

logger = logging.getLogger(__name__)

# Attempt to import torch for tensor detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Sink(ABC):
    """Base class for telemetry sinks with context manager support."""
    
    @abstractmethod
    def emit(self, event: TelemetryEvent) -> None:
        """Emit an event to this sink."""
        pass
    
    def flush(self) -> None:
        """Force buffer writes (if applicable)."""
        pass
    
    def close(self) -> None:
        """Close the sink (release resources)."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures close."""
        self.close()


class NoOpSink(Sink):
    """Sink that does nothing (for profile=off)."""
    
    def emit(self, event: TelemetryEvent) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class MemorySink(Sink):
    """
    Sink that keeps events in memory (circular buffer).
    
    Args:
        capacity: Maximum buffer capacity
    """
    
    def __init__(self, capacity: int = 512):
        self.buffer = RingBuffer[TelemetryEvent](capacity)
    
    def emit(self, event: TelemetryEvent) -> None:
        self.buffer.append(event)
    
    def snapshot(self) -> List[TelemetryEvent]:
        """Return copy of events in buffer."""
        return self.buffer.snapshot()
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = RingBuffer[TelemetryEvent](self.buffer.capacity)
    
    def flush(self) -> None:
        """Nothing to flush for memory sink."""
        pass

    def close(self) -> None:
        """Nothing to close for memory sink."""
        pass


class ConsoleSink(Sink):
    """
    Sink that prints events to console.
    
    Args:
        verbose: If True, print full payload
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def emit(self, event: TelemetryEvent) -> None:
        if self.verbose:
            print(f"[pyfolds] step={event.step_id} "
                  f"phase={event.phase} "
                  f"mode={event.mode} "
                  f"neuron={event.neuron_id} "
                  f"payload={event.payload}")
        else:
            print(f"[pyfolds] step={event.step_id} phase={event.phase}")
    
    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class JSONLinesSink(Sink):
    """
    Sink that writes events to JSON Lines file.
    
    Args:
        path: File path
        flush_every: Number of events before auto-flush (0 = never)
        truncate: If True, overwrite file on open; otherwise append
    
    Example:
        >>> with JSONLinesSink("telemetry.jsonl", truncate=True) as sink:
        ...     sink.emit(event)
        ... # File automatically closed
    """
    
    def __init__(self, path: Union[str, Path], flush_every: int = 10, truncate: bool = False):
        self.path = Path(path)
        self.flush_every = flush_every
        self.truncate = truncate
        self._count = 0
        self._file = None
    
    def __enter__(self):
        """Open file when entering context."""
        self._ensure_open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure close even with error."""
        self.close()
    
    def _ensure_open(self):
        """Open file if needed."""
        if self._file is None:
            mode = 'w' if self.truncate else 'a'
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, mode, encoding='utf-8')
    
    def _make_serializable(self, obj, _depth: int = 0, _max_depth: int = 10):
        """Convert objects to JSON-serializable formats."""
        if _depth > _max_depth:
            return str(obj)

        if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            return obj.detach().cpu().tolist()
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(item, _depth + 1, _max_depth) for item in obj]
        if isinstance(obj, dict):
            return {
                key: self._make_serializable(value, _depth + 1, _max_depth)
                for key, value in obj.items()
            }
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj

        # Fallback to string
        return str(obj)
    
    def emit(self, event: TelemetryEvent) -> None:
        """Emit event to file."""
        self._ensure_open()
        
        try:
            data = {
                'step_id': event.step_id,
                'phase': event.phase,
                'mode': event.mode,
                'neuron_id': event.neuron_id,
                'timestamp': event.timestamp,
                'wall_time': event.wall_time,
                'payload': event.payload
            }
            self._file.write(json.dumps(data) + '\n')
            
        except (TypeError, ValueError) as e:
            logger.warning(f"Serialization failed: {e}. Converting...")
            
            # Convert to serializable format
            serializable_data = {
                'step_id': event.step_id,
                'phase': event.phase,
                'mode': event.mode,
                'neuron_id': event.neuron_id,
                'timestamp': event.timestamp,
                'wall_time': event.wall_time,
                'payload': self._make_serializable(event.payload)
            }
            self._file.write(json.dumps(serializable_data) + '\n')
        
        self._count += 1
        if self.flush_every > 0 and self._count % self.flush_every == 0:
            self.flush()
    
    def flush(self) -> None:
        """Force write to disk."""
        if self._file is not None:
            self._file.flush()
    
    def close(self) -> None:
        """Close the file."""
        if self._file is not None:
            self.flush()
            self._file.close()
            self._file = None


class DistributorSink(Sink):
    """
    Sink that distributes events to multiple sinks.
    
    Useful for sending events to multiple destinations simultaneously:
    - MemorySink for MindBoard (real-time)
    - JSONLinesSink for MindAudit (persistence)
    - ConsoleSink for debug
    
    Args:
        sinks: List of sinks to distribute events to
    
    Example:
        >>> sink = DistributorSink([
        ...     MemorySink(1000),           # For MindBoard
        ...     JSONLinesSink("audit.jsonl") # For MindAudit
        ... ])
        >>> telem = TelemetryController(sink=sink)
    """
    
    def __init__(self, sinks: List[Sink]):
        self.sinks = sinks
    
    def emit(self, event: TelemetryEvent) -> None:
        """Distribute event to all sinks."""
        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception as e:
                logger.error(f"Error emitting to {type(sink).__name__}: {e}")
    
    def flush(self) -> None:
        """Flush all sinks that support it."""
        for sink in self.sinks:
            try:
                if hasattr(sink, 'flush'):
                    sink.flush()
            except Exception as e:
                logger.error(f"Error flushing {type(sink).__name__}: {e}")
    
    def close(self) -> None:
        """Close all sinks."""
        for sink in self.sinks:
            try:
                if hasattr(sink, 'close'):
                    sink.close()
            except Exception as e:
                logger.error(f"Error closing {type(sink).__name__}: {e}")