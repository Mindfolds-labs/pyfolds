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


def _base_full_cfg(*, n_synapses_per_dendrite: int, theta_init: float, lateral_strength: float) -> MPJRDConfig:
    return MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=n_synapses_per_dendrite,
        theta_init=theta_init,
        plasticity_mode="both",
        homeostasis_eta=0.1,
        lateral_strength=lateral_strength,
        inhibition_mode="both",
        refrac_mode="both",
        adaptation_enabled=True,
        adaptation_increment=0.6,
        adaptation_tau=50.0,
        backprop_enabled=True,
        backprop_delay=2.0,
        backprop_signal=0.3,
        wave_enabled=True,
        circadian_enabled=True,
        experimental_engram_enabled=True,
    )


def _create_retina(n_neurons: int) -> nn.ModuleList:
    cfg = _base_full_cfg(n_synapses_per_dendrite=4, theta_init=0.35, lateral_strength=0.1)
    return nn.ModuleList([MPJRDNeuronAdvanced(cfg) for _ in range(n_neurons)])


def _create_lgn(n_neurons: int) -> nn.ModuleList:
    cfg = _base_full_cfg(n_synapses_per_dendrite=4, theta_init=0.4, lateral_strength=0.5)
    return nn.ModuleList([MPJRDNeuronAdvanced(cfg) for _ in range(n_neurons)])


def _create_v1(n_neurons: int) -> nn.ModuleList:
    cfg = _base_full_cfg(n_synapses_per_dendrite=8, theta_init=0.45, lateral_strength=0.3)
    return nn.ModuleList([MPJRDNeuronAdvanced(cfg) for _ in range(n_neurons)])


def _create_it(n_neurons: int) -> nn.ModuleList:
    cfg = _base_full_cfg(n_synapses_per_dendrite=8, theta_init=0.5, lateral_strength=0.2)
    return nn.ModuleList([MPJRDNeuronAdvanced(cfg) for _ in range(n_neurons)])
