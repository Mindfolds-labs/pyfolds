import math

import torch

from pyfolds import MPJRDConfig
from pyfolds.advanced import (
    MPJRDNeuronAdvanced,
    analyze_mechanisms,
    compute_phase_amplitude_coupling,
    detect_envelope_events,
    extract_speech_envelope,
    latency_kernel,
    reset_phase_if_event,
)


def _tone(sr: int = 16000, secs: float = 0.1) -> torch.Tensor:
    t = torch.arange(int(sr * secs), dtype=torch.float32) / sr
    return torch.sin(2 * math.pi * 220.0 * t)


def test_extract_speech_envelope_hilbert_and_gammatone():
    audio = _tone()
    h = extract_speech_envelope(audio, 16000, method="hilbert")
    g = extract_speech_envelope(audio, 16000, method="gammatone")

    assert h["envelope"].shape == audio.shape
    assert g["envelope"].shape == audio.shape
    assert torch.all(h["onset_strength"] >= 0)
    assert "power" in h["modulation_spectrum"]


def test_detect_events_and_phase_reset():
    env = torch.tensor([0.0, 0.1, 0.2, 1.8, 1.9, 2.0])
    events = detect_envelope_events(env, threshold_scale=0.5)
    assert events["event_mask"].any()

    phase = torch.tensor([0.1, 0.2, 0.3])
    reset = reset_phase_if_event(phase, event_strength=torch.tensor(0.9), strength_threshold=0.5)
    assert torch.allclose(reset, torch.zeros_like(phase))


def test_pac_latency_and_analysis():
    phase = torch.linspace(-math.pi, math.pi, 128)
    amp = 1.0 + 0.5 * torch.cos(phase)
    pac = compute_phase_amplitude_coupling(phase, amp)
    assert float(pac["modulation_index"]) >= 0.0

    lat = latency_kernel(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert lat.shape == (2,)

    report = analyze_mechanisms(
        baseline={"mean_activity": 1.0, "stability": 0.9},
        with_mechanisms={"env": {"mean_activity": 1.2, "stability": 0.8}},
    )
    assert "env" in report


def test_wave_dynamics_keeps_baseline_when_speech_tracking_disabled():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        enable_speech_envelope_tracking=False,
        enable_phase_reset_on_audio_event=False,
        enable_cross_frequency_coupling=False,
        enable_spatial_latency_gradient=False,
    )
    neuron = MPJRDNeuronAdvanced(cfg)
    x = torch.randn(3, 2, 4)
    out = neuron(x, collect_stats=False)

    assert "speech_envelope" not in out
    assert "envelope_events" not in out
    assert "cross_frequency_coupling" not in out
    assert "spatial_latency_ms" not in out


def test_wave_dynamics_exposes_speech_payload_when_enabled():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        enable_speech_envelope_tracking=True,
        enable_phase_reset_on_audio_event=True,
        enable_cross_frequency_coupling=True,
        enable_spatial_latency_gradient=True,
    )
    neuron = MPJRDNeuronAdvanced(cfg)
    x = torch.randn(3, 2, 4)
    audio = _tone()
    out = neuron(x, collect_stats=False, audio=audio, audio_sample_rate=16000, neuron_coord=torch.tensor([1.0, 0.5]))

    assert "speech_envelope" in out
    assert "envelope_events" in out
    assert "cross_frequency_coupling" in out
    assert "spatial_latency_ms" in out
