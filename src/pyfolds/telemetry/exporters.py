"""Telemetry exporters."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from pyfolds.utils.compat import has_prometheus, has_tensorboard, require_prometheus, require_tensorboard

from .types import TelemetryEvent


class BaseExporter(ABC):
    @abstractmethod
    def export(self, events: Iterable[TelemetryEvent]) -> int:
        ...

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.flush()


class ConsoleExporter(BaseExporter):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def export(self, events: Iterable[TelemetryEvent]) -> int:
        count = 0
        for event in events:
            count += 1
            if self.verbose:
                print(f"[{event.event_type}] {event.source} step={event.step} payload={event.payload}")
            else:
                print(f"[{event.event_type}] {event.source} step={event.step}")
        return count


class CSVExporter(BaseExporter):
    def __init__(self, output_dir: str, filename: str = "telemetry.csv"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / filename
        self._file = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=["timestamp", "event_type", "source", "step", "epoch", "severity", "tags", "payload"])
        if self.path.stat().st_size == 0:
            self._writer.writeheader()

    def export(self, events: Iterable[TelemetryEvent]) -> int:
        count = 0
        for e in events:
            count += 1
            self._writer.writerow({
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "source": e.source,
                "step": e.step,
                "epoch": e.epoch,
                "severity": e.severity,
                "tags": json.dumps(e.tags or {}, ensure_ascii=False),
                "payload": json.dumps(e.payload, ensure_ascii=False, default=str),
            })
        return count

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self.flush()
        self._file.close()


class TensorBoardExporter(BaseExporter):
    def __init__(self, log_dir: str):
        if not has_tensorboard():
            require_tensorboard()
        SummaryWriter = require_tensorboard().SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)

    def export(self, events: Iterable[TelemetryEvent]) -> int:
        count = 0
        for e in events:
            count += 1
            if e.event_type == "loss" and isinstance(e.payload.get("loss"), (int, float)):
                self.writer.add_scalar("pyfolds/loss", float(e.payload["loss"]), e.step)
            if e.event_type == "spike" and isinstance(e.payload.get("rate"), (int, float)):
                self.writer.add_scalar("pyfolds/spikes/rate", float(e.payload["rate"]), e.step)
            if e.event_type.startswith("engram") and isinstance(e.payload.get("count"), (int, float)):
                self.writer.add_scalar("pyfolds/engrams/count", float(e.payload["count"]), e.step)
            if e.event_type == "forward_latency" and isinstance(e.payload.get("latency_ms"), (int, float)):
                self.writer.add_scalar("pyfolds/latency/forward_ms", float(e.payload["latency_ms"]), e.step)
        return count

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class PrometheusExporter(BaseExporter):
    def __init__(self, port: int = 8000):
        if not has_prometheus():
            require_prometheus()
        pc = require_prometheus()
        pc.start_http_server(port)
        self.events_total = pc.Counter("pyfolds_events_total", "Telemetry events", ["event_type"])
        self.forward_latency = pc.Histogram("pyfolds_forward_latency_ms", "Forward latency in ms")
        self.buffer_utilization = pc.Gauge("pyfolds_buffer_utilization", "Telemetry buffer utilization")

    def export(self, events: Iterable[TelemetryEvent]) -> int:
        count = 0
        for e in events:
            count += 1
            self.events_total.labels(event_type=e.event_type).inc()
            if e.event_type == "forward_latency" and isinstance(e.payload.get("latency_ms"), (int, float)):
                self.forward_latency.observe(float(e.payload["latency_ms"]))
        return count
