import copy

import pytest
import torch

from pyfolds import MPJRDConfig
from pyfolds.advanced import MPJRDNeuronAdvanced
from pyfolds.core.cognitive_controller import NetworkState, OrientationPolicy
from pyfolds.telemetry.controller import TelemetryConfig, TelemetryController


@pytest.fixture()
def cfg_base() -> dict:
    return {
        "n_dendrites": 2,
        "n_synapses_per_dendrite": 4,
        "theta_init": 0.1,
        "wave_enabled": True,
        "enable_speech_envelope_tracking": False,
        "enable_phase_reset_on_audio_event": False,
        "enable_cross_frequency_coupling": False,
        "enable_spatial_latency_gradient": False,
    }


@pytest.fixture()
def x_input() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(3, 2, 4)


def _clone_neuron(cfg: MPJRDConfig, seed: int) -> MPJRDNeuronAdvanced:
    torch.manual_seed(seed)
    return MPJRDNeuronAdvanced(cfg)


def _copy_state(src: MPJRDNeuronAdvanced, dst: MPJRDNeuronAdvanced) -> None:
    dst.load_state_dict(copy.deepcopy(src.state_dict()))


def test_experimental_toggles_off_preserve_baseline(cfg_base, x_input):
    cfg_default = MPJRDConfig(**{k: v for k, v in cfg_base.items() if not k.startswith("enable_")})
    cfg_explicit_off = MPJRDConfig(**cfg_base)

    neuron_default = _clone_neuron(cfg_default, seed=11)
    neuron_explicit_off = _clone_neuron(cfg_explicit_off, seed=12)
    _copy_state(neuron_default, neuron_explicit_off)

    out_default = neuron_default(x_input, collect_stats=False)
    out_explicit_off = neuron_explicit_off(x_input, collect_stats=False)

    for key in ("u", "spikes", "phase", "latency", "amplitude", "wave_real", "wave_imag"):
        assert torch.allclose(out_default[key], out_explicit_off[key], atol=1e-6, rtol=1e-5)


def test_phase_coherence_threshold_behavior(cfg_base, x_input):
    cfg_no_reset_dict = dict(cfg_base)
    cfg_no_reset_dict.update(enable_speech_envelope_tracking=True, enable_phase_reset_on_audio_event=True, phase_reset_threshold=10.0)
    cfg_with_reset_dict = dict(cfg_base)
    cfg_with_reset_dict.update(enable_speech_envelope_tracking=True, enable_phase_reset_on_audio_event=True, phase_reset_threshold=0.0)
    cfg_no_reset = MPJRDConfig(**cfg_no_reset_dict)
    cfg_with_reset = MPJRDConfig(**cfg_with_reset_dict)

    neuron_no_reset = _clone_neuron(cfg_no_reset, seed=21)
    neuron_with_reset = _clone_neuron(cfg_with_reset, seed=22)
    _copy_state(neuron_no_reset, neuron_with_reset)

    audio = torch.linspace(0, 1, 1600)
    out_no_reset = neuron_no_reset(x_input, collect_stats=False, audio=audio, audio_sample_rate=16000)
    out_with_reset = neuron_with_reset(x_input, collect_stats=False, audio=audio, audio_sample_rate=16000)

    mean_abs_phase_with_reset = out_with_reset["phase"].abs().mean().item()
    mean_abs_phase_no_reset = out_no_reset["phase"].abs().mean().item()
    assert mean_abs_phase_with_reset < mean_abs_phase_no_reset


def test_sleep_replay_offline_not_in_online_hotpath(cfg_base, x_input):
    cfg = MPJRDConfig(**cfg_base, replay_interval_steps=1)
    neuron = _clone_neuron(cfg, seed=31)

    calls = {"replay": 0}
    real_replay = neuron.run_replay_cycle

    def _count_replay():
        calls["replay"] += 1
        return real_replay()

    neuron.run_replay_cycle = _count_replay  # type: ignore[method-assign]
    neuron.update_network_state = lambda **_: None  # type: ignore[method-assign]

    neuron.network_state = NetworkState.ACTIVE
    neuron(x_input, collect_stats=True)
    assert calls["replay"] == 0

    neuron.network_state = NetworkState.MEMORY_REPLAY
    neuron._latest_policy = OrientationPolicy(
        current_mode=NetworkState.MEMORY_REPLAY,
        effective_eta=1.0,
        effective_attention_gain=1.0,
        effective_competition_gain=1.0,
        effective_replay_priority=1.0,
        effective_consolidation_rate=1.0,
        effective_decay_rate=1.0,
        sensory_excitability=1.0,
    )
    neuron(x_input, collect_stats=True)
    assert calls["replay"] == 1


def test_debug_traces_do_not_change_forward_outputs(cfg_base, x_input):
    cfg_no_debug = MPJRDConfig(**cfg_base, scientific_debug_stats=False)
    cfg_debug = MPJRDConfig(**cfg_base, scientific_debug_stats=True)

    neuron_no_debug = _clone_neuron(cfg_no_debug, seed=41)
    neuron_debug = _clone_neuron(cfg_debug, seed=42)
    _copy_state(neuron_no_debug, neuron_debug)

    out_no_debug = neuron_no_debug(x_input, collect_stats=True)
    out_debug = neuron_debug(x_input, collect_stats=True)

    for key in ("u", "u_raw", "spikes", "v_dend", "gated", "theta", "theta_eff"):
        assert torch.allclose(out_no_debug[key], out_debug[key], atol=1e-6, rtol=1e-5)


def test_telemetry_sleep_replay_events_and_resonance_metrics(cfg_base, x_input):
    cfg = MPJRDConfig(**cfg_base, replay_interval_steps=1)
    neuron = MPJRDNeuronAdvanced(cfg)
    neuron.telemetry = TelemetryController(TelemetryConfig(profile="heavy", sample_every=1))

    neuron.update_network_state = lambda **_: None  # type: ignore[method-assign]
    neuron.network_state = NetworkState.MEMORY_REPLAY
    neuron._latest_policy = OrientationPolicy(
        current_mode=NetworkState.MEMORY_REPLAY,
        effective_eta=1.0,
        effective_attention_gain=1.0,
        effective_competition_gain=1.0,
        effective_replay_priority=1.0,
        effective_consolidation_rate=1.0,
        effective_decay_rate=1.0,
        sensory_excitability=1.0,
    )

    out = neuron(x_input, collect_stats=True)
    neuron.sleep(duration=2.0)

    events = neuron.telemetry.sink.buffer.snapshot()
    by_type = {}
    for event in events:
        by_type[event.event_type] = by_type.get(event.event_type, 0) + 1

    sleep_replay_events = int(by_type.get("sleep", 0)) + int(out["network_state"] == NetworkState.MEMORY_REPLAY.value)
    assert sleep_replay_events >= 2

    engram = neuron.collect_engram_report()
    assert engram["resonance_by_dendrite"].shape == (cfg.n_dendrites,)
    assert torch.isfinite(engram["mean_resonance"]).all()
    assert torch.isfinite(engram["max_resonance"]).all()
