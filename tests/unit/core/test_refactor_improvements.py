import torch

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.homeostasis import HomeostasisController
from pyfolds.core.neuron import MPJRDNeuron


def test_homeostasis_stability_ratio_uses_configured_window() -> None:
    cfg = MPJRDConfig(homeostasis_stability_window=4, target_spike_rate=0.5, homeostasis_alpha=1.0)
    ctrl = HomeostasisController(cfg)

    # 2 estáveis, 2 instáveis no histórico circular
    for rate in [0.5, 0.5, 0.9, 0.9]:
        ctrl.update(rate)

    ratio = ctrl.stability_ratio(window=4)
    assert 0.49 <= ratio <= 0.51


def test_forward_handles_missing_optional_step_id_buffer() -> None:
    cfg = MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=3)
    neuron = MPJRDNeuron(cfg, enable_telemetry=False)
    neuron._buffers.pop("step_id", None)

    x = torch.zeros(2, cfg.n_dendrites, cfg.n_synapses_per_dendrite, device=neuron.theta.device)
    out = neuron.forward(x, dt=1.0)

    assert "spikes" in out
    assert neuron._safe_step_id() == 0


def test_circadian_gate_modulates_i_eta_within_bounds() -> None:
    cfg = MPJRDConfig(
        circadian_enabled=True,
        circadian_cycle_hours=1.0,
        circadian_plasticity_min=0.25,
        circadian_plasticity_max=1.25,
        i_eta=0.02,
    )
    neuron = MPJRDNeuron(cfg, enable_telemetry=False)
    x = torch.zeros(1, cfg.n_dendrites, cfg.n_synapses_per_dendrite, device=neuron.theta.device)

    neuron.forward(x, dt=5.0)
    assert cfg.i_eta * cfg.circadian_plasticity_min <= neuron.cfg.i_eta <= cfg.i_eta * cfg.circadian_plasticity_max
