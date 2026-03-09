from pyfolds.telemetry.collector import TelemetryCollector
from pyfolds.telemetry.integration import (
    attach_telemetry_to_engram_bank,
    attach_telemetry_to_neuron,
    attach_telemetry_to_specialization_engine,
)
from pyfolds.telemetry.types import TelemetryConfig


class Dummy:
    def __init__(self):
        self.t = 1
        self.r_hat = type("M", (), {"mean": lambda self: type("I", (), {"item": lambda self: 0.5})()})()
        self.engrams = [1]
        self.current_specialization = "vision"

    def forward(self, *args, **kwargs):
        return 1

    def consolidate(self, *args, **kwargs):
        return 1

    def update(self, *args, **kwargs):
        return 1


def test_attach_helpers_emit_events():
    c = TelemetryCollector(TelemetryConfig(auto_start=False))
    n = Dummy(); b = Dummy(); e = Dummy()
    attach_telemetry_to_neuron(n, c)
    attach_telemetry_to_engram_bank(b, c)
    attach_telemetry_to_specialization_engine(e, c)
    n.forward(); b.consolidate(step=1); e.update(step=1)
    assert c.buffer.size() >= 3
