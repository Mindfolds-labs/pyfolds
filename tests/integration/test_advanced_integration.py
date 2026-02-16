"""Teste de integração de múltiplos mixins avançados."""

import torch

from pyfolds.advanced import MPJRDNeuronAdvanced
from pyfolds.core.config import MPJRDConfig


def test_all_mixins_together_smoke():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=8,
        t_refrac_abs=2.0,
        t_refrac_rel=5.0,
        tau_pre=20.0,
        tau_post=20.0,
        adaptation_increment=0.1,
        u0=0.1,
        R0=1.0,
        backprop_enabled=True,
    )

    neuron = MPJRDNeuronAdvanced(cfg)
    x = torch.randn(4, 2, 8)

    out = neuron(x)

    expected_keys = [
        "spikes",
        "u",
        "v_dend",
        "theta",
        "r_hat",
        "u_adapted",
        "adaptation_current",
        "refrac_blocked",
        "theta_boost",
        "trace_pre_mean",
        "trace_post_mean",
        "dendrite_amplification",
    ]

    for key in expected_keys:
        assert key in out
