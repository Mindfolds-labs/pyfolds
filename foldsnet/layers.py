"""Construção das camadas biológicas da FOLDSNet."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn

if importlib.util.find_spec("pyfolds") is None:
    src_path = Path(__file__).resolve().parents[1] / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

from pyfolds.advanced import MPJRDNeuronAdvanced
from pyfolds.core.config import MPJRDConfig


@dataclass(frozen=True)
class LayerProfile:
    n_dendrites: int
    n_synapses_per_dendrite: int
    theta_init: float
    lateral_strength: float


_LAYER_PROFILES: dict[str, LayerProfile] = {
    "retina": LayerProfile(n_dendrites=4, n_synapses_per_dendrite=4, theta_init=0.35, lateral_strength=0.1),
    "lgn": LayerProfile(n_dendrites=4, n_synapses_per_dendrite=4, theta_init=0.4, lateral_strength=0.5),
    "v1": LayerProfile(n_dendrites=4, n_synapses_per_dendrite=8, theta_init=0.45, lateral_strength=0.3),
    "it": LayerProfile(n_dendrites=4, n_synapses_per_dendrite=8, theta_init=0.5, lateral_strength=0.2),
}


def _cfg_from_profile(profile: LayerProfile) -> MPJRDConfig:
    return MPJRDConfig(
        n_dendrites=profile.n_dendrites,
        n_synapses_per_dendrite=profile.n_synapses_per_dendrite,
        theta_init=profile.theta_init,
        plasticity_mode="both",
        homeostasis_eta=0.1,
        lateral_strength=profile.lateral_strength,
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


def _create_layer(stage: str, n_neurons: int) -> nn.ModuleList:
    profile = _LAYER_PROFILES[stage]
    cfg = _cfg_from_profile(profile)
    return nn.ModuleList([MPJRDNeuronAdvanced(cfg) for _ in range(n_neurons)])


def _create_retina(n_neurons: int) -> nn.ModuleList:
    return _create_layer("retina", n_neurons)


def _create_lgn(n_neurons: int) -> nn.ModuleList:
    return _create_layer("lgn", n_neurons)


def _create_v1(n_neurons: int) -> nn.ModuleList:
    return _create_layer("v1", n_neurons)


def _create_it(n_neurons: int) -> nn.ModuleList:
    return _create_layer("it", n_neurons)
