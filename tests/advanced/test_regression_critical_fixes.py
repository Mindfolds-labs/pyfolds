import time

import torch

from pyfolds.advanced import MPJRDWaveNeuronAdvanced
from pyfolds.advanced.inhibition import InhibitionLayer
from pyfolds.advanced.noetic_model import NoeticCore
from pyfolds.advanced.refractory import RefractoryMixin
from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.utils.types import LearningMode


class RefractoryOnlyNeuron(RefractoryMixin, MPJRDNeuron):
    def __init__(self, cfg: MPJRDConfig):
        super().__init__(cfg)
        self._init_refractory(
            t_refrac_abs=cfg.t_refrac_abs,
            t_refrac_rel=cfg.t_refrac_rel,
            refrac_rel_strength=cfg.refrac_rel_strength,
        )


def test_wave_advanced_forward_initializes_mixins() -> None:
    cfg = MPJRDConfig(wave_enabled=True)
    neuron = MPJRDWaveNeuronAdvanced(cfg)
    x = torch.zeros(2, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
    out = neuron.forward(x)
    assert "spikes" in out


def test_refractory_mixin_is_self_sufficient_for_time_counter() -> None:
    cfg = MPJRDConfig(
        theta_init=0.5,
        t_refrac_abs=2.0,
        t_refrac_rel=2.0,
        refrac_rel_strength=0.0,
    )
    neuron = RefractoryOnlyNeuron(cfg)
    x = torch.ones(1, cfg.n_dendrites, cfg.n_synapses_per_dendrite)

    s1 = float(neuron.forward(x, dt=1.0)["spikes"].item())
    s2 = float(neuron.forward(x, dt=1.0)["spikes"].item())
    s3 = float(neuron.forward(x, dt=1.0)["spikes"].item())
    s4 = float(neuron.forward(x, dt=1.0)["spikes"].item())

    assert s1 == 1.0
    assert s2 == 0.0
    assert s3 == 0.0
    assert s4 == 1.0


def test_noetic_sleep_updates_base_neuromodulator() -> None:
    noetic = NoeticCore(MPJRDConfig())
    noetic.sleep()
    assert noetic.neuromodulator.current_mode == LearningMode.SLEEP


def test_inhibition_vectorized_initialization_benchmark() -> None:
    n_exc = 1000
    n_inh = 250

    def old_lateral() -> torch.Tensor:
        positions = torch.arange(n_exc, dtype=torch.float32)
        kernel = torch.zeros(n_exc, n_exc)
        for i in range(n_exc):
            distances = torch.abs(positions - i)
            kernel[i] = torch.exp(-distances**2 / (2 * 5.0**2))
        kernel.fill_diagonal_(0)
        return kernel

    def old_i2e() -> torch.Tensor:
        w = torch.zeros(n_inh, n_exc)
        for i in range(n_inh):
            center = (i / n_inh) * n_exc
            positions = torch.arange(n_exc, dtype=torch.float32)
            distances = torch.abs(positions - center)
            sigma = n_exc / 10.0
            w[i] = torch.exp(-distances**2 / (2 * sigma**2)) * 0.8
        return w

    t0 = time.perf_counter()
    _ = old_lateral()
    _ = old_i2e()
    old_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    layer = InhibitionLayer(n_excitatory=n_exc, n_inhibitory=n_inh)
    new_time = time.perf_counter() - t1

    assert layer.lateral_kernel.shape == (n_exc, n_exc)
    assert layer.W_I2E.shape == (n_inh, n_exc)
    assert new_time < old_time
