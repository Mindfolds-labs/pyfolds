from pyfolds.telemetry import TelemetryCollector, TelemetryConfig, make_loss_event

cfg = TelemetryConfig(enable_tensorboard=True, tensorboard_log_dir="outputs/tensorboard")
collector = TelemetryCollector(cfg)
for step in range(10):
    collector.emit(make_loss_event("train", step=step, loss=1.0/(step+1)))
collector.stop()
