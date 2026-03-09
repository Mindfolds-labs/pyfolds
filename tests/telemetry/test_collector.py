from pyfolds.telemetry.collector import TelemetryCollector
from pyfolds.telemetry.exporters import BaseExporter
from pyfolds.telemetry.events import make_loss_event
from pyfolds.telemetry.types import TelemetryConfig


class FailingExporter(BaseExporter):
    def export(self, events):
        raise RuntimeError("boom")


def test_collector_start_stop_idempotent():
    c = TelemetryCollector(TelemetryConfig(auto_start=False, flush_interval_seconds=0.01))
    c.start(); c.start()
    c.stop(); c.stop()


def test_collector_exporter_failure_isolated():
    c = TelemetryCollector(TelemetryConfig(auto_start=False), exporters=[FailingExporter()])
    c.start()
    c.emit(make_loss_event("train", 1, 0.1))
    c.force_flush()
    assert c.get_stats().exporter_failures >= 1
    c.stop()
