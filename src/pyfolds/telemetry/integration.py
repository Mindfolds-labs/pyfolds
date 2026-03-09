"""Helpers to attach telemetry to existing components with low coupling."""

from __future__ import annotations

from functools import wraps

from .collector import TelemetryCollector
from .events import make_engram_event, make_specialization_event, make_spike_event


def _wrap_method(obj, method_name: str, wrapper):
    original = getattr(obj, method_name, None)
    if original is None or getattr(original, "_telemetry_wrapped", False):
        return

    @wraps(original)
    def inner(*args, **kwargs):
        result = original(*args, **kwargs)
        wrapper(result, *args, **kwargs)
        return result

    inner._telemetry_wrapped = True
    setattr(obj, method_name, inner)


def attach_telemetry_to_neuron(neuron, collector: TelemetryCollector):
    def after_forward(result, *args, **kwargs):
        step = int(getattr(neuron, "t", 0))
        rate = float(getattr(neuron, "r_hat", 0.0).mean().item()) if hasattr(getattr(neuron, "r_hat", 0), "mean") else 0.0
        collector.emit(make_spike_event(type(neuron).__name__, step, rate))

    _wrap_method(neuron, "forward", after_forward)


def attach_telemetry_to_engram_bank(bank, collector: TelemetryCollector):
    def after_consolidate(result, *args, **kwargs):
        step = int(kwargs.get("step", 0))
        count = len(getattr(bank, "engrams", [])) if hasattr(bank, "engrams") else None
        collector.emit(make_engram_event(type(bank).__name__, step, action="consolidated", count=count))

    _wrap_method(bank, "consolidate", after_consolidate)


def attach_telemetry_to_specialization_engine(engine, collector: TelemetryCollector):
    def after_update(result, *args, **kwargs):
        step = int(kwargs.get("step", 0))
        spec = getattr(engine, "current_specialization", "unknown")
        collector.emit(make_specialization_event(type(engine).__name__, step, specialization=str(spec)))

    for method in ("update", "forward"):
        _wrap_method(engine, method, after_update)
