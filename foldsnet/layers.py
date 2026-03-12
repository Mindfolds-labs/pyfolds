"""Construção das camadas biológicas da FOLDSNet."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch.nn as nn

if importlib.util.find_spec("pyfolds") is None:
    src_path = Path(__file__).resolve().parents[1] / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

from pyfolds.advanced import MPJRDNeuronAdvanced
from pyfolds.core.config import MPJRDConfig


def _create_retina(n_neurons: int) -> nn.ModuleList:
    """Cria camada Retina com inibição lateral fraca e sem mecanismos avançados."""
    cfg = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=4,
        theta_init=0.35,
        plasticity_mode="stdp",
        homeostasis_eta=0.1,
        lateral_strength=0.1,
        refrac_mode="absolute",
        adaptation_enabled=False,
        backprop_enabled=False,
        wave_enabled=False,
        circadian_enabled=False,
        experimental_engram_enabled=False,
    )
    return nn.ModuleList([MPJRDNeuronAdvanced(cfg) for _ in range(n_neurons)])


def _create_lgn(n_neurons: int) -> nn.ModuleList:
    """Cria camada LGN com inibição lateral forte para realce de bordas."""
    cfg = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=4,
        theta_init=0.4,
        plasticity_mode="stdp",
        homeostasis_eta=0.1,
        lateral_strength=0.5,
        inhibition_mode="lateral",
        refrac_mode="both",
        adaptation_enabled=False,
        backprop_enabled=False,
        wave_enabled=False,
        circadian_enabled=False,
        experimental_engram_enabled=False,
    )
    return nn.ModuleList([MPJRDNeuronAdvanced(cfg) for _ in range(n_neurons)])


def _create_v1(n_neurons: int) -> nn.ModuleList:
    """Cria camada V1 (simples + complexas)."""
    half = n_neurons // 2

    simple_cfg = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=8,
        theta_init=0.45,
        plasticity_mode="stdp",
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

    complex_cfg = MPJRDConfig(
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

    layers = [MPJRDNeuronAdvanced(simple_cfg) for _ in range(half)]
    layers.extend(MPJRDNeuronAdvanced(complex_cfg) for _ in range(n_neurons - half))
    return nn.ModuleList(layers)


def _create_it(n_neurons: int) -> nn.ModuleList:
    """Cria camada IT com todos os mecanismos ativos."""
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
    return nn.ModuleList([MPJRDNeuronAdvanced(cfg) for _ in range(n_neurons)])
