"""Construção das camadas biológicas da FOLDSNet."""

from __future__ import annotations

import torch.nn as nn

from pyfolds.advanced import MPJRDNeuronAdvanced
from pyfolds.core.config import MPJRDConfig


def _create_retina(n_retina: int) -> nn.ModuleList:
    """Cria camada Retina com mecanismos biológicos restritos."""
    cfg = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=4,
        theta_init=0.35,
        plasticity_mode="stdp_only",
        homeostasis_eta=0.1,
        lateral_strength=0.1,
        refrac_mode="abs_only",
        adaptation_enabled=False,
        backprop_enabled=False,
        wave_enabled=False,
        circadian_enabled=False,
        experimental_engram_enabled=False,
    )
    return nn.ModuleList(MPJRDNeuronAdvanced(cfg) for _ in range(n_retina))


def _create_lgn(n_lgn: int) -> nn.ModuleList:
    """Cria camada LGN com inibição lateral forte."""
    cfg = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=4,
        theta_init=0.4,
        plasticity_mode="stdp_only",
        homeostasis_eta=0.1,
        lateral_strength=0.5,
        inhibition_mode="lateral_only",
        refrac_mode="both",
        adaptation_enabled=False,
        backprop_enabled=False,
        wave_enabled=False,
        circadian_enabled=False,
        experimental_engram_enabled=False,
    )
    return nn.ModuleList(MPJRDNeuronAdvanced(cfg) for _ in range(n_lgn))


def _create_v1(n_v1: int) -> nn.ModuleList:
    """Cria camada V1 (metade simples, metade complexa)."""
    neurons = nn.ModuleList()
    simple = n_v1 // 2
    complex_count = n_v1 - simple

    cfg_simple = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=8,
        theta_init=0.45,
        plasticity_mode="stdp_only",
        homeostasis_eta=0.15,
        lateral_strength=0.3,
        inhibition_mode="both",
        refrac_mode="both",
        adaptation_enabled=True,
        adaptation_increment=0.8,
        adaptation_tau=50.0,
        backprop_enabled=False,
        wave_enabled=False,
        circadian_enabled=False,
        experimental_engram_enabled=False,
    )
    cfg_complex = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=8,
        theta_init=0.5,
        plasticity_mode="both",
        homeostasis_eta=0.15,
        lateral_strength=0.3,
        inhibition_mode="both",
        refrac_mode="both",
        adaptation_enabled=True,
        adaptation_increment=0.8,
        adaptation_tau=50.0,
        backprop_enabled=False,
        wave_enabled=True,
        circadian_enabled=False,
        experimental_engram_enabled=False,
    )

    for _ in range(simple):
        neurons.append(MPJRDNeuronAdvanced(cfg_simple))
    for _ in range(complex_count):
        neurons.append(MPJRDNeuronAdvanced(cfg_complex))
    return neurons


def _create_it(n_it: int) -> nn.ModuleList:
    """Cria camada IT com todos os mecanismos habilitados."""
    cfg = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=8,
        theta_init=0.5,
        plasticity_mode="both",
        homeostasis_eta=0.2,
        lateral_strength=0.2,
        inhibition_mode="both",
        refrac_mode="both",
        adaptation_enabled=True,
        adaptation_increment=0.6,
        backprop_enabled=True,
        backprop_delay=2.0,
        backprop_signal=0.3,
        wave_enabled=True,
        circadian_enabled=True,
        experimental_engram_enabled=True,
    )
    return nn.ModuleList(MPJRDNeuronAdvanced(cfg) for _ in range(n_it))
