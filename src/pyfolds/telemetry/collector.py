"""Asynchronous telemetry collector."""

from __future__ import annotations

import threading
import time
from typing import Iterable

from .buffer import RingBufferThreadSafe
from .exporters import BaseExporter, ConsoleExporter, CSVExporter, PrometheusExporter, TensorBoardExporter
from .types import TelemetryConfig, TelemetryEvent, TelemetryStats


class TelemetryCollector:
    def __init__(self, config: TelemetryConfig | None = None, exporters: Iterable[BaseExporter] | None = None):
        self.config = config or TelemetryConfig()
        self.buffer = RingBufferThreadSafe[TelemetryEvent](
            capacity=self.config.buffer_size,
            drop_oldest_on_overflow=self.config.drop_oldest_on_overflow,
        )
        self.stats = TelemetryStats()
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._exporters: list[BaseExporter] = list(exporters or [])
        self._bootstrap_exporters()
        if self.config.auto_start:
            self.start()

    def _bootstrap_exporters(self) -> None:
        if self.config.enable_console:
            self._exporters.append(ConsoleExporter())
        if self.config.enable_csv:
            self._exporters.append(CSVExporter(self.config.csv_output_dir))
        if self.config.enable_tensorboard:
            self._exporters.append(TensorBoardExporter(self.config.tensorboard_log_dir))
        if self.config.enable_prometheus:
            self._exporters.append(PrometheusExporter(self.config.prometheus_port))

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self.stats.started_at = time.time()
            self._stop_event.clear()
            self._worker = threading.Thread(target=self._run_worker, daemon=True, name="pyfolds-telemetry")
            self._worker.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self.force_flush()
        for exporter in list(self._exporters):
            try:
                exporter.close()
            except Exception:
                self.stats.exporter_failures += 1

    def emit(self, event: TelemetryEvent) -> None:
        if not self.config.event_allowed(event.event_type):
            return
        pushed = self.buffer.push(event)
        self.stats.events_collected += 1
        if not pushed:
            self.stats.events_dropped += 1

    def _run_worker(self) -> None:
        while not self._stop_event.wait(self.config.flush_interval_seconds):
            self.force_flush()

    def force_flush(self) -> None:
        events = self.buffer.drain(self.config.max_batch_size)
        while events:
            for exporter in list(self._exporters):
                try:
                    exported = exporter.export(events)
                    self.stats.events_exported += exported
                except Exception:
                    self.stats.exporter_failures += 1
            events = self.buffer.drain(self.config.max_batch_size)
        for exporter in list(self._exporters):
            try:
                exporter.flush()
            except Exception:
                self.stats.exporter_failures += 1
        self.stats.events_dropped = self.buffer.dropped_events_count()
        self.stats.buffer_utilization = self.buffer.size() / float(self.config.buffer_size)

    def get_stats(self) -> TelemetryStats:
        self.stats.events_dropped = self.buffer.dropped_events_count()
        self.stats.buffer_utilization = self.buffer.size() / float(self.config.buffer_size)
        return self.stats

    def register_exporter(self, exporter: BaseExporter) -> None:
        self._exporters.append(exporter)

    def unregister_exporter(self, exporter: BaseExporter) -> None:
        if exporter in self._exporters:
            self._exporters.remove(exporter)

    def __enter__(self) -> "TelemetryCollector":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
