from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from pyfolds.advanced import MPJRDNeuronAdvanced
from pyfolds.core.config import MPJRDConfig

from training.config.mnist import RunConfig
from training.models.contracts import FOLDSNetAdapter, MPJRDWrapper, ModelMetadata


def _build_mpjrd_config(config: RunConfig, device: torch.device) -> MPJRDConfig:
    mp = config.mpjrd
    base = config.base
    return MPJRDConfig(
        n_dendrites=mp.n_dendrites,
        n_synapses_per_dendrite=mp.n_synapses_per_dendrite,
        n_min=1,
        n_max=100,
        w_scale=3.5,
        i_eta=base.lr,
        i_gamma=0.99,
        i_min=-20.0,
        i_max=50.0,
        u0=0.2,
        R0=1.0,
        U=0.3,
        tau_fac=100.0,
        tau_rec=800.0,
        theta_init=mp.threshold,
        theta_min=0.2,
        theta_max=4.0,
        target_spike_rate=0.2,
        homeostasis_eta=0.2,
        dead_neuron_threshold=0.01,
        activity_threshold=0.01,
        dendrite_integration_mode="nmda_shunting",
        dendrite_gain=2.0,
        backprop_enabled=not mp.disable_backprop,
        backprop_delay=2.0,
        backprop_signal=0.5,
        adaptation_enabled=not mp.disable_sfa,
        adaptation_increment=0.8,
        adaptation_decay=0.99,
        adaptation_max=5.0,
        adaptation_tau=50.0,
        refrac_mode="both" if not mp.disable_refratario else "none",
        t_refrac_abs=2.0,
        t_refrac_rel=5.0,
        refrac_rel_strength=3.0,
        inhibition_mode="both" if not mp.disable_inibicao else "none",
        lateral_strength=0.3,
        feedback_strength=0.5,
        n_excitatory=mp.hidden,
        n_inhibitory=max(1, mp.hidden // 4),
        plasticity_mode="both" if not mp.disable_stdp else "none",
        tau_pre=20.0,
        tau_post=20.0,
        A_plus=0.01,
        A_minus=0.012,
        neuromod_mode="surprise",
        sup_k=3.0,
        wave_enabled=not mp.disable_wave,
        circadian_enabled=not mp.disable_circadian,
        experimental_engram_enabled=not mp.disable_engram,
        enable_speech_envelope_tracking=not mp.disable_speech,
        plastic=True,
        defer_updates=True,
        dt=1.0,
        device=str(device),
    )


def build_model(config: RunConfig, device: torch.device):
    base = config.base
    if base.model == "mpjrd":
        mp_cfg = _build_mpjrd_config(config, device)
        neuron = MPJRDNeuronAdvanced(cfg=mp_cfg)
        model = MPJRDWrapper(neuron, config.mpjrd.n_dendrites, config.mpjrd.n_synapses_per_dendrite).to(device)
        metadata = ModelMetadata(
            family="mpjrd",
            config=asdict(config.mpjrd),
            supports_learning_mode=True,
            supports_sleep=True,
        )
        return model, metadata, mp_cfg

    if base.model == "foldsnet":
        if importlib.util.find_spec("foldsnet") is None:
            root_path = Path(__file__).resolve().parents[2]
            if str(root_path) not in sys.path:
                sys.path.insert(0, str(root_path))
        from foldsnet.factory import create_foldsnet

        folds_cfg = config.foldsnet
        raw_model = create_foldsnet(folds_cfg.variant, folds_cfg.dataset).to(device)
        model = FOLDSNetAdapter(raw_model, folds_cfg.variant, folds_cfg.dataset).to(device)
        metadata = ModelMetadata(
            family="foldsnet",
            config=asdict(folds_cfg),
            supports_learning_mode=False,
            supports_sleep=False,
        )
        return model, metadata, None

    raise ValueError(f"Modelo inválido: {base.model}")


def extract_logits(output: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    logits = output[0] if isinstance(output, tuple) else output
    if isinstance(logits, dict):
        return logits.get("logits", logits.get("spikes", torch.zeros(batch_size, 10, device=device)))
    return logits
