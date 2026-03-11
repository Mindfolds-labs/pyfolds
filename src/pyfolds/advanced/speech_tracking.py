"""Mecanismos opcionais para neural speech tracking."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional

import torch


def _as_tensor_1d(audio: torch.Tensor | list[float]) -> torch.Tensor:
    x = torch.as_tensor(audio, dtype=torch.float32)
    if x.ndim != 1:
        raise ValueError("audio must be 1-D")
    return x


def _analytic_signal_hilbert(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    Xf = torch.fft.fft(x)
    h = torch.zeros(n, dtype=torch.float32, device=x.device)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return torch.fft.ifft(Xf * h.to(Xf.dtype))


def _erb_space(f_min: float, f_max: float, n_filters: int) -> torch.Tensor:
    erb = lambda f: 21.4 * torch.log10(4.37e-3 * f + 1.0)
    inv_erb = lambda e: (10 ** (e / 21.4) - 1.0) / 4.37e-3
    e0 = erb(torch.tensor(f_min))
    e1 = erb(torch.tensor(f_max))
    pts = torch.linspace(float(e0), float(e1), n_filters)
    return inv_erb(pts)


def _gammatone_envelope(audio: torch.Tensor, sample_rate: int, n_filters: int = 16) -> torch.Tensor:
    n = audio.shape[0]
    spec = torch.fft.rfft(audio)
    freqs = torch.fft.rfftfreq(n, d=1.0 / float(sample_rate)).to(audio.device)
    centers = _erb_space(80.0, min(sample_rate / 2.0 - 1.0, 7800.0), n_filters).to(audio.device)

    envelopes = []
    for cf in centers:
        bw = 1.019 * 24.7 * (4.37 * cf / 1000.0 + 1.0)
        filt = torch.exp(-0.5 * ((freqs - cf) / (bw + 1e-6)) ** 2)
        band = torch.fft.irfft(spec * filt.to(spec.dtype), n=n)
        env = torch.abs(_analytic_signal_hilbert(band))
        envelopes.append(env)
    return torch.stack(envelopes, dim=0).mean(dim=0)


def _modulation_spectrum(envelope: torch.Tensor, sample_rate: int) -> Dict[str, torch.Tensor]:
    env = envelope - envelope.mean()
    mod_spec = torch.fft.rfft(env)
    mod_freqs = torch.fft.rfftfreq(env.shape[0], d=1.0 / float(sample_rate))
    return {
        "frequency_hz": mod_freqs,
        "power": (mod_spec.abs() ** 2) / max(1, env.shape[0]),
    }


def extract_speech_envelope(
    audio: torch.Tensor | list[float],
    sample_rate: int,
    method: str = "gammatone",
) -> Dict[str, Any]:
    x = _as_tensor_1d(audio)
    if method == "hilbert":
        envelope = torch.abs(_analytic_signal_hilbert(x))
    elif method == "gammatone":
        envelope = _gammatone_envelope(x, sample_rate=sample_rate)
    else:
        raise ValueError("method must be 'hilbert' or 'gammatone'")

    d_env = torch.diff(envelope, prepend=envelope[:1])
    onset_strength = torch.relu(d_env)
    return {
        "envelope": envelope,
        "onset_strength": onset_strength,
        "modulation_spectrum": _modulation_spectrum(envelope, sample_rate),
    }


def detect_envelope_events(envelope: torch.Tensor, threshold_scale: float = 1.5) -> Dict[str, Any]:
    env = _as_tensor_1d(envelope)
    onset_strength = torch.relu(torch.diff(env, prepend=env[:1]))
    thr = onset_strength.mean() + threshold_scale * onset_strength.std(unbiased=False)
    event_mask = onset_strength >= thr
    onset_times = torch.nonzero(event_mask, as_tuple=False).squeeze(-1)
    return {
        "onset_times": onset_times,
        "onset_strength": onset_strength,
        "event_mask": event_mask,
    }


def reset_phase_if_event(
    phase: torch.Tensor,
    event_strength: torch.Tensor | float,
    strength_threshold: float = 0.25,
    target_phase: float = 0.0,
) -> torch.Tensor:
    strength = torch.as_tensor(event_strength, dtype=phase.dtype, device=phase.device)
    if strength.ndim == 0:
        strength = strength.expand_as(phase)
    mask = strength >= strength_threshold
    target = torch.full_like(phase, float(target_phase))
    return torch.where(mask, target, phase)


def compute_phase_amplitude_coupling(phase_theta: torch.Tensor, amp_gamma: torch.Tensor, n_bins: int = 18) -> Dict[str, torch.Tensor]:
    phase = torch.as_tensor(phase_theta, dtype=torch.float32).reshape(-1)
    amp = torch.as_tensor(amp_gamma, dtype=torch.float32).reshape(-1)
    if phase.numel() != amp.numel():
        raise ValueError("phase_theta and amp_gamma must have same number of elements")

    bins = torch.linspace(-math.pi, math.pi, n_bins + 1, device=phase.device)
    inds = torch.bucketize(phase.clamp(-math.pi, math.pi), bins) - 1
    inds = inds.clamp(0, n_bins - 1)
    mean_amp = torch.zeros(n_bins, device=phase.device)
    for i in range(n_bins):
        mask = inds == i
        mean_amp[i] = amp[mask].mean() if mask.any() else 0.0

    prob = mean_amp + 1e-8
    prob = prob / prob.sum()
    uniform = torch.full_like(prob, 1.0 / n_bins)
    kl = torch.sum(prob * torch.log(prob / uniform))
    modulation_index = kl / math.log(n_bins)
    return {"modulation_index": modulation_index, "phase_bins_mean_amplitude": mean_amp}


def latency_kernel(neuron_coord: torch.Tensor, max_latency_ms: float = 100.0, spatial_scale: float = 1.0) -> torch.Tensor:
    coord = torch.as_tensor(neuron_coord, dtype=torch.float32)
    if coord.ndim == 1:
        dist = torch.linalg.norm(coord)
    else:
        dist = torch.linalg.norm(coord, dim=-1)
    return max_latency_ms * torch.tanh(dist / max(spatial_scale, 1e-6))


def analyze_mechanisms(
    baseline: Mapping[str, float],
    with_mechanisms: Mapping[str, Mapping[str, float]],
    connectivity_baseline: Optional[torch.Tensor] = None,
    connectivity_variants: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    for mech, metrics in with_mechanisms.items():
        t0 = time.perf_counter()
        delta_activity = metrics.get("mean_activity", baseline.get("mean_activity", 0.0)) - baseline.get("mean_activity", 0.0)
        delta_stability = metrics.get("stability", baseline.get("stability", 0.0)) - baseline.get("stability", 0.0)
        conn_delta = 0.0
        if connectivity_baseline is not None and connectivity_variants and mech in connectivity_variants:
            conn_delta = float((connectivity_variants[mech] - connectivity_baseline).abs().mean().item())
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        report[mech] = {
            "impact_mean_activity": float(delta_activity),
            "impact_stability": float(delta_stability),
            "computational_cost_ms": float(elapsed_ms),
            "active_connectivity_change": float(conn_delta),
        }
    return report
