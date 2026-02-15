"""Tipos de payload para telemetria"""

from typing import TypedDict, Optional


class ForwardPayload(TypedDict, total=False):
    """Payload tipado para fase forward."""
    spike_rate: float
    theta: float
    r_hat: float
    v_dend_mean: float
    u_mean: float
    saturation_ratio: float
    N_mean: float
    I_mean: float
    W_mean: float
    duration_ms: float


class CommitPayload(TypedDict, total=False):
    """Payload tipado para fase commit."""
    post_rate: float
    R: float
    delta_N_mean: float
    delta_I_mean: float
    synapses_updated: int
    duration_ms: float


class SleepPayload(TypedDict, total=False):
    """Payload tipado para fase sleep."""
    duration: float
    N_mean_before: float
    N_mean_after: float
    I_mean_before: float
    I_mean_after: float