"""Teste de integração: ciclo completo de aprendizagem com circadian."""

import warnings

import pytest
import torch

from pyfolds import MPJRDConfig
from pyfolds.advanced import MPJRDNeuronAdvanced
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.utils.types import LearningMode


@pytest.fixture
def advanced_neuron() -> MPJRDNeuronAdvanced:
    cfg = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=8,
        theta_init=0.5,
        plastic=True,
        wave_enabled=True,
        circadian_enabled=True,
        circadian_auto_mode=False,
    )
    return MPJRDNeuronAdvanced(cfg)


def test_full_online_batch_sleep_cycle(advanced_neuron: MPJRDNeuronAdvanced) -> None:
    """ONLINE → BATCH → SLEEP → métricas consistentes."""
    neuron = advanced_neuron
    x = torch.randn(16, 4, 8)

    neuron.set_mode(LearningMode.ONLINE)
    for _ in range(10):
        out = neuron(x, reward=0.6, dt=1.0)
    assert out["mode"] == "online"

    neuron.set_mode(LearningMode.BATCH)
    for _ in range(5):
        neuron(x, reward=0.4, dt=1.0)
    neuron.apply_plasticity(dt=1.0, reward=0.4)

    neuron.set_mode(LearningMode.SLEEP)
    neuron.sleep(duration=60.0)

    metrics = neuron.get_metrics()
    assert metrics["theta"] > 0
    assert 0.0 <= metrics["saturation_ratio"] <= 1.0
    assert metrics["mode"] == "sleep"
    assert not metrics["has_pending_updates"]


def test_circadian_gate_reduces_plasticity_in_pm(advanced_neuron: MPJRDNeuronAdvanced) -> None:
    """PM deve reduzir learning rate via gate circadiano."""
    neuron = advanced_neuron

    neuron.circadian_phase.fill_(180.0)
    x = torch.randn(8, 4, 8)
    out = neuron(x, reward=0.5, dt=1.0)

    gate_pm = out.get("circadian_plasticity_gate", None)
    if gate_pm is not None:
        assert gate_pm < 1.0

    neuron.circadian_phase.fill_(0.0)
    out_am = neuron(x, reward=0.5, dt=1.0)
    gate_am = out_am.get("circadian_plasticity_gate", None)

    if gate_am is not None and gate_pm is not None:
        assert gate_am > gate_pm


def test_homeostasis_stability_ratio_real_values(advanced_neuron: MPJRDNeuronAdvanced) -> None:
    """stability_ratio deve retornar valor entre 0 e 1, não binário."""
    neuron = advanced_neuron
    x = torch.randn(8, 4, 8)

    for _ in range(200):
        neuron(x, dt=1.0)

    ratio = neuron.homeostasis.stability_ratio(window=100)
    assert 0.0 <= ratio <= 1.0
    assert not (ratio != ratio)


def test_no_register_buffer_warnings_on_init() -> None:
    """__init__ não deve emitir warnings de register_buffer duplicado."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=8)
        _ = MPJRDNeuron(cfg)

    buffer_warnings = [x for x in w if "register_buffer" in str(x.message).lower()]
    assert len(buffer_warnings) == 0
