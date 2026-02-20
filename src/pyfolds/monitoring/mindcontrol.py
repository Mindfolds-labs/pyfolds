"""MindControl runtime injection engine for closed-loop parameter mutation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, Optional
import weakref

from ..telemetry.events import TelemetryEvent
from ..telemetry.sinks import Sink


@dataclass(frozen=True)
class MutationCommand:
    """Single runtime mutation request."""

    name: str
    value: Any


class MindControl:
    """Closed-loop controller that injects runtime mutations into neurons."""

    def __init__(self, decision_fn: Optional[Callable[[TelemetryEvent], Optional[MutationCommand]]] = None):
        self._decision_fn = decision_fn
        self._neurons: "weakref.WeakSet[Any]" = weakref.WeakSet()
        self._lock = Lock()

    def register_neuron(self, neuron: Any) -> None:
        """Register a neuron that accepts ``queue_runtime_injection`` calls."""
        if not hasattr(neuron, "queue_runtime_injection"):
            raise TypeError("Neuron must implement queue_runtime_injection(name, value)")
        with self._lock:
            self._neurons.add(neuron)

    def inject_parameter(self, name: str, value: Any) -> int:
        """Queue mutation for all registered neurons without blocking forward path."""
        with self._lock:
            neurons = list(self._neurons)

        for neuron in neurons:
            neuron.queue_runtime_injection(name, value)
        return len(neurons)

    def on_telemetry_event(self, event: TelemetryEvent) -> None:
        """Process telemetry events and optionally emit a mutation command."""
        if event.phase != "commit" or self._decision_fn is None:
            return

        command = self._decision_fn(event)
        if command is None:
            return

        self.inject_parameter(command.name, command.value)


class MindControlSink(Sink):
    """Telemetry sink that forwards COMMIT events to a MindControl instance."""

    def __init__(self, controller: MindControl):
        self.controller = controller

    def emit(self, event: TelemetryEvent) -> None:
        self.controller.on_telemetry_event(event)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
