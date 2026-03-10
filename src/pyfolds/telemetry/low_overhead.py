"""Low-overhead telemetry collector."""

from __future__ import annotations

from collections import deque
import threading
from typing import Deque, Iterable

from .exporters import BaseExporter
from .types import TelemetryConfig, TelemetryEvent, TelemetryStats


class LowOverheadMetricsCollector:
    """Telemetry collector optimized for training hot-paths.

    Parameters
    ----------
    config : TelemetryConfig | None
        Collector configuration.
    exporters : Iterable[BaseExporter] | None
        Export backends that receive flushed events.

    Returns
    -------
    None

    Examples
    --------
    >>> collector = LowOverheadMetricsCollector(TelemetryConfig(auto_start=False))
    >>> collector.emit(TelemetryEvent(time.time(), "forward", "train", 1, {}))
    >>> collector.force_flush()
    """

    def __init__(self, config: TelemetryConfig | None = None, exporters: Iterable[BaseExporter] | None = None):
        self.config = config or TelemetryConfig()
        self.stats = TelemetryStats()
        self._ring: Deque[TelemetryEvent] = deque(maxlen=self.config.buffer_size)
        self._dropped = 0
        self._ring_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._exporters: list[BaseExporter] = list(exporters or [])
        if self.config.auto_start:
            self.start()

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run_worker, daemon=True, name="pyfolds-telemetry-low")
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self.force_flush()

    def emit(self, event: TelemetryEvent) -> None:
        if not self.config.event_allowed(event.event_type):
            return
        with self._ring_lock:
            before = len(self._ring)
            self._ring.append(event)
            if len(self._ring) == before and before == self.config.buffer_size:
                self._dropped += 1
        self.stats.events_collected += 1

    def _run_worker(self) -> None:
        while not self._stop_event.wait(self.config.flush_interval_seconds):
            self.force_flush()

    def force_flush(self) -> None:
        with self._ring_lock:
            batch = list(self._ring)
            self._ring.clear()
        if not batch:
            return
        for exporter in list(self._exporters):
            try:
                self.stats.events_exported += exporter.export(batch)
                exporter.flush()
            except Exception:
                self.stats.exporter_failures += 1
        self.stats.events_dropped = self._dropped
        self.stats.buffer_utilization = len(self._ring) / float(self.config.buffer_size)
