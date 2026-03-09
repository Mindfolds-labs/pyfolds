"""Factories for standardized telemetry events."""

from __future__ import annotations

import time
from typing import Any, Callable

from .types import TelemetryEvent


def _event(event_type: str, source: str, step: int, payload: dict[str, Any], **kwargs: Any) -> TelemetryEvent:
    return TelemetryEvent(timestamp=time.time(), event_type=event_type, source=source, step=step, payload=payload, **kwargs)


# new api

def make_spike_event(source: str, step: int, rate: float, neuron_id: str | None = None) -> TelemetryEvent:
    return _event("spike", source, step, {"rate": float(rate), "neuron_id": neuron_id})


def make_loss_event(source: str, step: int, loss: float, epoch: int | None = None) -> TelemetryEvent:
    return _event("loss", source, step, {"loss": float(loss)}, epoch=epoch)


def make_engram_event(source: str, step: int, action: str, count: int | None = None, **extra: Any) -> TelemetryEvent:
    payload = {"action": action, "count": count}
    payload.update(extra)
    return _event(f"engram_{action}", source, step, payload)


def make_specialization_event(source: str, step: int, specialization: str, score: float | None = None) -> TelemetryEvent:
    return _event("specialization_update", source, step, {"specialization": specialization, "score": score})


def make_checkpoint_event(source: str, step: int, action: str, path: str) -> TelemetryEvent:
    return _event(f"checkpoint_{action}", source, step, {"path": path})


def make_latency_event(source: str, step: int, latency_ms: float, phase: str = "forward") -> TelemetryEvent:
    return _event("forward_latency", source, step, {"latency_ms": float(latency_ms), "phase": phase})


# legacy api compatibility

def forward_event(step_id: int, mode: str, neuron_id: str | None = None, **payload: Any) -> TelemetryEvent:
    payload.setdefault("neuron_id", neuron_id)
    return _event("forward", mode, step_id, payload)


def commit_event(step_id: int, mode: str, neuron_id: str | None = None, **payload: Any) -> TelemetryEvent:
    payload.setdefault("neuron_id", neuron_id)
    return _event("commit", mode, step_id, payload)


def sleep_event(step_id: int, mode: str, neuron_id: str | None = None, **payload: Any) -> TelemetryEvent:
    payload.setdefault("neuron_id", neuron_id)
    return _event("sleep", mode, step_id, payload)


def forward_event_lazy(step_id: int, mode: str, payload_fn: Callable[[], dict[str, Any]], neuron_id: str | None = None) -> TelemetryEvent:
    payload = payload_fn(); payload.setdefault("neuron_id", neuron_id)
    return _event("forward", mode, step_id, payload)


def commit_event_lazy(step_id: int, mode: str, payload_fn: Callable[[], dict[str, Any]], neuron_id: str | None = None) -> TelemetryEvent:
    payload = payload_fn(); payload.setdefault("neuron_id", neuron_id)
    return _event("commit", mode, step_id, payload)


def sleep_event_lazy(step_id: int, mode: str, payload_fn: Callable[[], dict[str, Any]], neuron_id: str | None = None) -> TelemetryEvent:
    payload = payload_fn(); payload.setdefault("neuron_id", neuron_id)
    return _event("sleep", mode, step_id, payload)
