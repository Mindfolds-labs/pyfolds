"""Smoke tests for the public import surface exposed by ``pyfolds``.

These tests protect downstream projects that depend on top-level imports from
``pyfolds`` and ensure key classes/functions are still importable.
"""

import warnings

from packaging.version import Version

import pyfolds


def test_public_all_exports_are_importable():
    missing = [name for name in pyfolds.__all__ if not hasattr(pyfolds, name)]
    assert missing == [], f"Export(s) ausentes no módulo pyfolds: {missing}"


def test_v2_surface_is_canonical_and_instantiable():
    cfg = pyfolds.NeuronConfig(n_dendrites=2)
    neuron = pyfolds.MPJRDNeuron(cfg)
    layer = pyfolds.AdaptiveNeuronLayer(1, cfg)
    network = pyfolds.SpikingNetwork()
    network.add_layer("input", layer)

    assert neuron is not None
    assert layer is not None
    assert network is not None


def test_v1_aliases_emit_deprecation_warning_and_match_v2_targets_until_2_0():
    removal_version = Version("2.0.0")
    current_version = Version(pyfolds.__version__)

    if current_version >= removal_version:
        assert not hasattr(pyfolds, "MPJRDConfig")
        assert not hasattr(pyfolds, "MPJRDLayer")
        assert not hasattr(pyfolds, "MPJRDNetwork")
        return

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy_cfg = pyfolds.MPJRDConfig
        legacy_layer = pyfolds.MPJRDLayer
        legacy_network = pyfolds.MPJRDNetwork

    assert legacy_cfg is pyfolds.NeuronConfig
    assert legacy_layer is pyfolds.AdaptiveNeuronLayer
    assert legacy_network is pyfolds.SpikingNetwork

    assert len(caught) == 3
    assert all(issubclass(item.category, DeprecationWarning) for item in caught)


def test_telemetry_controller_basic_flow():
    sink = pyfolds.MemorySink(capacity=8)
    controller = pyfolds.TelemetryController(
        cfg=pyfolds.TelemetryConfig(profile="heavy"), sink=sink
    )

    controller.emit(pyfolds.forward_event(step_id=0, mode="test", value=1.0))

    events = controller.snapshot()
    assert len(events) >= 1
