from pyfolds.telemetry import TelemetryCollector, TelemetryConfig, make_latency_event

cfg = TelemetryConfig(enable_prometheus=True, prometheus_port=8010)
collector = TelemetryCollector(cfg)
collector.emit(make_latency_event("inference", step=1, latency_ms=3.4))
collector.stop()
