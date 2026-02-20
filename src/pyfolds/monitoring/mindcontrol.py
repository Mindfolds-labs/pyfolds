"""MindControl runtime injection engine for closed-loop parameter mutation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, Iterable, Optional
import queue
import weakref

from ..telemetry.events import TelemetryEvent
from ..telemetry.sinks import Sink


@dataclass(frozen=True)
class MutationCommand:
    """Single runtime mutation request."""

    name: str
    value: Any


class MutationQueue:
    """Thread-safe queue for deferred mutation application in neuron boundaries."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()

    def inject(self, param_name: str, value: Any) -> None:
        self._queue.put((param_name, value))

    def fetch_all(self) -> list[tuple[str, Any]]:
        mutations: list[tuple[str, Any]] = []
        while not self._queue.empty():
            try:
                mutations.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return mutations


class MindControlEngine:
    """Decision engine that consumes telemetry and queues safe runtime mutations."""

    def __init__(self) -> None:
        self.mutation_queues: Dict[str, MutationQueue] = {}
        self.safety_bounds = {
            "activity_threshold": (0.01, 0.99),
            "learning_rate": (1e-6, 1e-1),
            "target_spike_rate": (0.0, 1.0),
        }

    def register_neuron(self, neuron_id: str) -> MutationQueue:
        mq = MutationQueue()
        self.mutation_queues[neuron_id] = mq
        return mq

    def _bounded(self, name: str, value: Any) -> Any:
        bounds = self.safety_bounds.get(name)
        if bounds is None:
            return value

        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return value

        low, high = bounds
        return max(low, min(high, scalar))

    def analyze_and_react(self, event_data: Dict[str, Any]) -> None:
        """Analyze telemetry payload and enqueue bounded mutations."""
        try:
            neuron_id = str(event_data.get("neuron_id", "")).strip()
            if not neuron_id or neuron_id not in self.mutation_queues:
                return

            spike_rate = event_data.get("spike_rate", event_data.get("post_rate", 0.0))
            try:
                spike_rate = float(spike_rate)
            except (TypeError, ValueError):
                return

            if spike_rate < 0.05:
                self.mutation_queues[neuron_id].inject(
                    "activity_threshold",
                    self._bounded("activity_threshold", 0.1),
                )
            elif spike_rate > 0.95:
                self.mutation_queues[neuron_id].inject(
                    "activity_threshold",
                    self._bounded("activity_threshold", 0.8),
                )
        except Exception:
            # Zero-crash policy: the control plane must never interrupt the core loop.
            return


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

    def _iter_neurons(self) -> Iterable[Any]:
        with self._lock:
            return list(self._neurons)

    def inject_parameter(self, name: str, value: Any) -> int:
        """Queue mutation for all registered neurons without blocking forward path."""
        neurons = list(self._iter_neurons())
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

    def __init__(self, controller: Optional[MindControl] = None, engine: Optional[MindControlEngine] = None):
        if controller is None and engine is None:
            raise ValueError("MindControlSink requires either controller or engine")
        self.controller = controller
        self.engine = engine

    def emit(self, event: TelemetryEvent) -> None:
        if event.phase != "commit":
            return
        if self.controller is not None:
            self.controller.on_telemetry_event(event)
        if self.engine is not None:
            payload = dict(event.payload)
            if event.neuron_id and "neuron_id" not in payload:
                payload["neuron_id"] = event.neuron_id
            self.engine.analyze_and_react(payload)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
