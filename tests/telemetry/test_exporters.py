from pyfolds.telemetry.events import make_latency_event, make_loss_event
from pyfolds.telemetry.exporters import CSVExporter, ConsoleExporter, PrometheusExporter
from pyfolds.utils.compat import has_prometheus


def test_csv_exporter_writes(tmp_path):
    e = CSVExporter(str(tmp_path))
    e.export([make_loss_event("train", 1, 0.2)])
    e.flush(); e.close()
    text = (tmp_path / "telemetry.csv").read_text(encoding="utf-8")
    assert "event_type" in text and "loss" in text


def test_console_exporter_payloads():
    e = ConsoleExporter(verbose=True)
    assert e.export([make_loss_event("x", 1, 1.0), make_latency_event("x", 2, 2.0)]) == 2


def test_prometheus_exporter_optional():
    if not has_prometheus():
        return
    e = PrometheusExporter(port=8020)
    assert e.export([make_latency_event("x", 1, 1.2)]) == 1
