from pyfolds.telemetry import TelemetryCollector, TelemetryConfig, make_loss_event

collector = TelemetryCollector(TelemetryConfig(enable_console=True))
collector.emit(make_loss_event("trainer", step=1, loss=0.42))
collector.stop()
