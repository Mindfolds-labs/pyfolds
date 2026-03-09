"""Telemetry typed structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass(frozen=True)
class TelemetryEvent:
    timestamp: float
    event_type: str
    source: str
    step: int
    payload: dict[str, Any] = field(default_factory=dict)
    epoch: int | None = None
    severity: str | None = None
    tags: dict[str, str] | None = None

    # legacy compatibility properties
    @property
    def step_id(self) -> int:
        return self.step

    @property
    def phase(self) -> str:
        return self.event_type

    @property
    def mode(self) -> str:
        return self.source

    @property
    def neuron_id(self) -> str | None:
        return self.payload.get("neuron_id") if isinstance(self.payload, dict) else None

    @property
    def wall_time(self) -> float:
        return self.timestamp


@dataclass
class TelemetryStats:
    events_collected: int = 0
    events_exported: int = 0
    events_dropped: int = 0
    exporter_failures: int = 0
    started_at: float = field(default_factory=time.time)
    buffer_utilization: float = 0.0

    @property
    def uptime_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)


@dataclass
class TelemetryConfig:
    buffer_size: int = 4096
    flush_interval_seconds: float = 1.0
    enable_console: bool = False
    enable_csv: bool = False
    enable_tensorboard: bool = False
    enable_prometheus: bool = False
    csv_output_dir: str = "outputs/telemetry"
    tensorboard_log_dir: str = "outputs/tensorboard"
    prometheus_port: int = 8000
    enabled_event_types: set[str] | None = None
    max_batch_size: int = 256
    drop_oldest_on_overflow: bool = True
    auto_start: bool = True
    strict_mode: bool = False

    def event_allowed(self, event_type: str) -> bool:
        return self.enabled_event_types is None or event_type in self.enabled_event_types


from typing import TypedDict

class ForwardPayload(TypedDict, total=False):
    spike_rate: float
    theta: float

class CommitPayload(TypedDict, total=False):
    post_rate: float
    R: float

class SleepPayload(TypedDict, total=False):
    duration: float
