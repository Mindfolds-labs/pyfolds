"""Smoke tests for the public import surface exposed by ``pyfolds``.

These tests protect downstream projects that depend on top-level imports from
``pyfolds`` and ensure key classes/functions are still importable.
"""

import pyfolds


def test_public_all_exports_are_importable():
    missing = [name for name in pyfolds.__all__ if not hasattr(pyfolds, name)]
    assert missing == [], f"Export(s) ausentes no módulo pyfolds: {missing}"


def test_core_objects_can_be_instantiated():
    cfg = pyfolds.MPJRDConfig(n_dendrites=2)
    neuron = pyfolds.MPJRDNeuron(cfg)
    layer = pyfolds.MPJRDLayer(1, cfg)
    network = pyfolds.MPJRDNetwork()
    network.add_layer("input", layer)

    assert neuron is not None
    assert layer is not None
    assert network is not None


def test_telemetry_controller_basic_flow():
    sink = pyfolds.MemorySink(capacity=8)
    controller = pyfolds.TelemetryController(
        cfg=pyfolds.TelemetryConfig(profile="heavy"), sink=sink
    )

    controller.emit(pyfolds.forward_event(step_id=0, mode="test", value=1.0))

    events = controller.snapshot()
    assert len(events) >= 1
