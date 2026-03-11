import torch

from pyfolds import MPJRDConfig
from pyfolds.advanced import MPJRDNeuronAdvanced


def test_advanced_neuron_wave_outputs_when_enabled():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
    )
    neuron = MPJRDNeuronAdvanced(cfg)
    x = torch.randn(3, 2, 4)
    out = neuron(x, collect_stats=False)

    assert "wave_real" in out
    assert "wave_imag" in out
    assert out["wave_real"].shape == (3,)


def test_advanced_neuron_has_no_wave_outputs_when_disabled():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=False,
    )
    neuron = MPJRDNeuronAdvanced(cfg)
    x = torch.randn(3, 2, 4)
    out = neuron(x, collect_stats=False)

    assert "wave_real" not in out
    assert "phase" not in out


def test_advanced_neuron_exposes_coherence_metrics_only_when_experimental_toggle_is_enabled():
    cfg_off = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        enable_experimental_coherence_metrics=False,
    )
    neuron_off = MPJRDNeuronAdvanced(cfg_off)
    x = torch.randn(3, 2, 4)
    out_off = neuron_off(x, collect_stats=False)

    assert "coherence_score" not in out_off
    assert "coherence_band" not in out_off

    cfg_on = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        enable_experimental_coherence_metrics=True,
        coherence_low_threshold=0.2,
        coherence_high_threshold=0.8,
    )
    neuron_on = MPJRDNeuronAdvanced(cfg_on)
    out_on = neuron_on(x, collect_stats=False)

    assert "coherence_score" in out_on
    assert "coherence_band" in out_on
    assert out_on["coherence_band"] in {"low", "medium", "high"}


def test_debug_oscillation_traces_adds_only_observability_payload():
    x = torch.randn(3, 2, 4)
    base_cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        random_seed=123,
    )
    debug_cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        random_seed=123,
        debug_oscillation_traces=True,
    )

    torch.manual_seed(123)
    base_neuron = MPJRDNeuronAdvanced(base_cfg)
    base_out = base_neuron(x, collect_stats=False)

    torch.manual_seed(123)
    debug_neuron = MPJRDNeuronAdvanced(debug_cfg)
    debug_out = debug_neuron(x, collect_stats=False)

    for key in ("spikes", "u", "phase", "phase_sync", "wave_real", "wave_imag", "wave_complex"):
        assert torch.allclose(base_out[key], debug_out[key])

    assert "debug_oscillation_traces" in debug_out
    assert "debug_oscillation_traces" not in base_out
