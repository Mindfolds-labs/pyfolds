"""Property-based tests for MPJRD neuron invariants."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron


@st.composite
def neuron_inputs(draw: st.DrawFn) -> tuple[int, int, int, float]:
    batch = draw(st.integers(min_value=1, max_value=8))
    dendrites = draw(st.integers(min_value=1, max_value=6))
    synapses = draw(st.integers(min_value=1, max_value=8))
    value = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return batch, dendrites, synapses, value


@settings(max_examples=30, deadline=None)
@given(neuron_inputs())
def test_spike_output_stays_in_valid_range(params: tuple[int, int, int, float]) -> None:
    batch, dendrites, synapses, value = params
    cfg = MPJRDConfig(n_dendrites=dendrites, n_synapses_per_dendrite=synapses, device="cpu")
    neuron = MPJRDNeuron(cfg)
    x = torch.full((batch, dendrites, synapses), value)

    output = neuron(x)
    spikes = output["spikes"]

    assert spikes.dtype.is_floating_point
    assert torch.all((spikes == 0.0) | (spikes == 1.0))


@settings(max_examples=30, deadline=None)
@given(st.integers(min_value=1, max_value=6), st.integers(min_value=1, max_value=8))
def test_membrane_potential_is_finite(dendrites: int, synapses: int) -> None:
    cfg = MPJRDConfig(n_dendrites=dendrites, n_synapses_per_dendrite=synapses, device="cpu")
    neuron = MPJRDNeuron(cfg)
    x = torch.randn(16, dendrites, synapses)

    output = neuron(x)

    assert torch.isfinite(output["u"]).all()
    assert torch.isfinite(output["v_dend"]).all()


@settings(max_examples=20, deadline=None)
@given(st.integers(min_value=1, max_value=4), st.integers(min_value=1, max_value=4))
def test_inference_is_deterministic_with_fixed_seed(dendrites: int, synapses: int) -> None:
    seed = 1234
    cfg = MPJRDConfig(n_dendrites=dendrites, n_synapses_per_dendrite=synapses, device="cpu")

    torch.manual_seed(seed)
    neuron_a = MPJRDNeuron(cfg)
    x = torch.randn(6, dendrites, synapses)
    out_a = neuron_a(x)

    torch.manual_seed(seed)
    neuron_b = MPJRDNeuron(cfg)
    out_b = neuron_b(x)

    assert torch.equal(out_a["spikes"], out_b["spikes"])
    assert torch.allclose(out_a["u"], out_b["u"], atol=1e-6)
