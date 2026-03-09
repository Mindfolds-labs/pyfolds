"""Tests for MPJRDSynapse."""

from unittest import mock

import pytest
import torch
import pyfolds
from pyfolds.utils.types import LearningMode


class TestMPJRDSynapse:
    """Test synapse."""
    
    def test_initialization(self, tiny_config):
        """Test initialization."""
        from pyfolds.core import MPJRDSynapse
        
        syn = MPJRDSynapse(tiny_config)
        assert syn.N.numel() == 1
        assert syn.I.numel() == 1
    
    def test_ltp(self, tiny_config):
        """Test LTP."""
        from pyfolds.core import MPJRDSynapse
        
        syn = MPJRDSynapse(tiny_config, init_n=0)
        initial_n = syn.N.item()
        
        syn.I.fill_(tiny_config.i_ltp_th + 1)
        syn.update(
            pre_rate=torch.tensor([1.0]),
            post_rate=torch.tensor([1.0]),
            R=torch.tensor([1.0])
        )
        
        assert syn.N.item() == initial_n + 1

    def test_update_respects_plastic_flag(self, tiny_config):
        """When plasticity is disabled, synaptic state must remain unchanged."""
        from pyfolds import NeuronConfig
        from pyfolds.core import MPJRDSynapse

        cfg = NeuronConfig(**{**tiny_config.to_dict(), "plastic": False})
        syn = MPJRDSynapse(cfg, init_n=3)

        before = syn.get_state()
        syn.update(pre_rate=torch.tensor([1.0]), post_rate=torch.tensor([1.0]), R=torch.tensor([1.0]))
        after = syn.get_state()

        assert torch.equal(before["N"], after["N"])
        assert torch.equal(before["I"], after["I"])
        assert torch.equal(before["eligibility"], after["eligibility"])

    def test_update_filters_inactive_pre_synaptic_activity(self, tiny_config):
        """Pre-synaptic activity below threshold should not change I/eligibility."""
        from pyfolds import NeuronConfig
        from pyfolds.core import MPJRDSynapse

        cfg = NeuronConfig(**{**tiny_config.to_dict(), "activity_threshold": 0.5, "i_ltp_th": 100.0})
        syn = MPJRDSynapse(cfg, init_n=2)

        syn.update(
            pre_rate=torch.tensor([0.2, 0.3]),
            post_rate=torch.tensor([1.0]),
            R=torch.tensor([1.0]),
        )

        assert syn.I.item() == pytest.approx(0.0)
        assert syn.eligibility.item() == pytest.approx(0.0)

    def test_update_uses_absolute_dt_and_mode_multiplier(self, tiny_config):
        """Negative dt should be handled as absolute value in plasticity update."""
        from pyfolds.core import MPJRDSynapse

        syn = MPJRDSynapse(tiny_config, init_n=2)
        syn.update(
            pre_rate=torch.tensor([1.0]),
            post_rate=torch.tensor([1.0]),
            R=torch.tensor([1.0]),
            dt=-2.0,
            mode=LearningMode.BATCH,
        )

        expected_delta = (
            tiny_config.i_eta
            * tiny_config.A_plus
            * torch.tanh(torch.tensor(tiny_config.neuromod_scale)).item()
            * LearningMode.BATCH.learning_rate_multiplier
            * 2.0
        )
        assert syn.I.item() == pytest.approx(expected_delta, rel=1e-5)

    def test_saturation_recovery_disables_protection_after_timeout(self, tiny_config):
        """Protection mode should auto-disable after recovery timeout."""
        from pyfolds import NeuronConfig
        from pyfolds.core import MPJRDSynapse

        cfg = NeuronConfig(**{**tiny_config.to_dict(), "saturation_recovery_time": 10.0, "i_ltp_th": 100.0})
        syn = MPJRDSynapse(cfg, init_n=cfg.n_max)
        syn.protection.fill_(True)
        syn.sat_time.fill_(9.0)

        syn.update(
            pre_rate=torch.tensor([0.0]),
            post_rate=torch.tensor([0.0]),
            R=torch.tensor([0.0]),
            dt=1.0,
        )

        assert syn.protection.item() is False
        assert syn.sat_time.item() == pytest.approx(0.0)


    def test_negative_neuromodulation_promotes_ltd(self, tiny_config):
        """Negative neuromodulation should bias update toward LTD."""
        from pyfolds import NeuronConfig
        from pyfolds.core import MPJRDSynapse

        cfg = NeuronConfig(**{
            **tiny_config.to_dict(),
            "hebbian_ltd_ratio": 1.0,
            "i_ltp_th": 100.0,
            "i_ltd_th": -100.0,
            "ltd_threshold_saturated": -120.0,
        })
        syn = MPJRDSynapse(cfg, init_n=3)

        syn.update(
            pre_rate=torch.tensor([1.0]),
            post_rate=torch.tensor([1.0]),
            R=torch.tensor([-1.0]),
        )

        assert syn.I.item() <= 0.0

    def test_consolidate_transfers_eligibility_and_decays_internal_potential(self, tiny_config):
        """Consolidation should move eligibility into N and decay I."""
        from pyfolds import NeuronConfig
        from pyfolds.core import MPJRDSynapse

        cfg = NeuronConfig(**{**tiny_config.to_dict(), "consolidation_rate": 1.0, "i_decay_sleep": 0.5})
        syn = MPJRDSynapse(cfg, init_n=5)
        syn.eligibility.fill_(2.0)
        syn.I.fill_(4.0)

        syn.consolidate(dt=1.0)

        assert syn.N.item() == 7
        assert syn.eligibility.item() == pytest.approx(0.0)
        assert syn.I.item() == pytest.approx(2.0)


    def test_update_supports_explicit_hebbian_ltd_component(self, tiny_config):
        """Explicit LTD term should reduce/update internal potential when configured."""
        from pyfolds import NeuronConfig
        from pyfolds.core import MPJRDSynapse

        cfg = NeuronConfig(**{
            **tiny_config.to_dict(),
            "hebbian_ltd_ratio": 1.0,
            "i_ltp_th": 100.0,
            "i_ltd_th": -100.0,
            "ltd_threshold_saturated": -120.0,
        })
        syn = MPJRDSynapse(cfg, init_n=3)

        syn.update(
            pre_rate=torch.tensor([1.0]),
            post_rate=torch.tensor([0.0]),
            R=torch.tensor([1.0]),
        )

        assert syn.I.item() < 0.0
        assert syn.eligibility.item() < 0.0


    def test_consolidate_performs_distributed_sync_when_enabled(self, tiny_config):
        """Consolidation should all-reduce plasticity buffers in distributed mode."""
        from pyfolds import NeuronConfig
        from pyfolds.core import MPJRDSynapse

        cfg = NeuronConfig(**{**tiny_config.to_dict(), "distributed_sync_on_consolidate": True})
        syn = MPJRDSynapse(cfg, init_n=4)
        syn.eligibility.fill_(2.0)

        fake_dist = mock.Mock()
        fake_dist.is_available.return_value = True
        fake_dist.is_initialized.return_value = True
        fake_dist.get_world_size.return_value = 2
        fake_dist.ReduceOp.SUM = object()

        with mock.patch.object(torch, "distributed", fake_dist):
            syn.consolidate(dt=1.0)

        assert fake_dist.all_reduce.call_count == 5

    def test_consolidate_skips_distributed_sync_when_disabled(self, tiny_config):
        """Distributed all-reduce should be skipped when config disables it."""
        from pyfolds import NeuronConfig
        from pyfolds.core import MPJRDSynapse

        cfg = NeuronConfig(**{**tiny_config.to_dict(), "distributed_sync_on_consolidate": False})
        syn = MPJRDSynapse(cfg, init_n=4)

        fake_dist = mock.Mock()
        fake_dist.is_available.return_value = True
        fake_dist.is_initialized.return_value = True
        fake_dist.get_world_size.return_value = 2

        with mock.patch.object(torch, "distributed", fake_dist):
            syn.consolidate(dt=1.0)

        fake_dist.all_reduce.assert_not_called()


def test_logN_backward_compat():
    """Modo logN deve manter lei de peso original."""
    from pyfolds import NeuronConfig
    from pyfolds.core import MPJRDSynapse

    cfg = NeuronConfig(weight_quantization="logN", n_max=31, w_scale=5.0)
    syn = MPJRDSynapse(cfg, init_n=0)

    for n in range(0, 32):
        syn.N.fill_(n)
        expected = torch.log2(torch.tensor(1.0 + n)) / cfg.w_scale
        assert syn.W.item() == pytest.approx(expected.item(), abs=1e-12)


def test_uniformW_constant_step():
    """uniformW deve ter degrau constante de peso entre níveis consecutivos."""
    from pyfolds import NeuronConfig
    from pyfolds.core import MPJRDSynapse

    cfg = NeuronConfig(weight_quantization="uniformW", n_levels=32, n_max=31, w_scale=5.0)
    syn = MPJRDSynapse(cfg, init_n=0)

    steps = []
    for l in range(cfg.n_levels - 1):
        syn.L.fill_(l)
        w0 = syn.W.item()
        syn.L.fill_(l + 1)
        w1 = syn.W.item()
        steps.append(w1 - w0)

    reference = steps[0]
    for step in steps[1:]:
        assert step == pytest.approx(reference, abs=1e-7)


def test_uniformW_bounds():
    """Atualizações em uniformW devem clamp L em [0, levels-1]."""
    from pyfolds import NeuronConfig
    from pyfolds.core import MPJRDSynapse

    cfg = NeuronConfig(weight_quantization="uniformW", n_levels=32, i_ltp_th=1.0, i_ltd_th=-1.0)
    syn = MPJRDSynapse(cfg, init_n=cfg.n_max)
    syn.L.fill_(cfg.n_levels - 1)

    for _ in range(5):
        syn.I.fill_(cfg.i_ltp_th + 1.0)
        syn.update(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]))
    assert syn.L.item() == cfg.n_levels - 1

    syn.protection.fill_(False)
    for _ in range(40):
        syn.I.fill_(cfg.i_ltd_th - 1.0)
        syn.update(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0]))
    assert syn.L.item() == 0


def test_uniformW_monotonic():
    """Peso deve crescer monotonicamente com L no modo uniformW."""
    from pyfolds import NeuronConfig
    from pyfolds.core import MPJRDSynapse

    cfg = NeuronConfig(weight_quantization="uniformW", n_levels=32)
    syn = MPJRDSynapse(cfg, init_n=0)

    for l in range(cfg.n_levels - 1):
        syn.L.fill_(l)
        w0 = syn.W.item()
        syn.L.fill_(l + 1)
        w1 = syn.W.item()
        assert w1 > w0


def test_uniformW_wmax():
    """Último nível de uniformW deve atingir w_max."""
    from pyfolds import NeuronConfig
    from pyfolds.core import MPJRDSynapse

    cfg = NeuronConfig(weight_quantization="uniformW", n_levels=32)
    syn = MPJRDSynapse(cfg, init_n=0)

    syn.L.fill_(cfg.n_levels - 1)
    assert syn.W.item() == pytest.approx(cfg.w_max)


def test_uniformW_updates_change_L_and_keep_N_as_telemetry():
    """No modo uniformW, plasticidade altera L e mantém N sincronizado."""
    from pyfolds import NeuronConfig
    from pyfolds.core import MPJRDSynapse

    cfg = NeuronConfig(weight_quantization="uniformW", n_levels=32, n_max=31)
    syn = MPJRDSynapse(cfg, init_n=0)

    syn.L.fill_(10)
    syn.N.fill_(0)
    syn.I.fill_(cfg.i_ltp_th + 1.0)
    syn.update(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]))

    assert syn.L.item() == 11
    expected_n = round((11 / (cfg.n_levels - 1)) * cfg.n_max)
    assert syn.N.item() == expected_n


def test_synapse_state_roundtrip_preserves_quantized_state():
    """Serialização deve preservar estado discreto e pesos em roundtrip."""
    from pyfolds import NeuronConfig
    from pyfolds.core import MPJRDSynapse

    cfg = NeuronConfig(weight_quantization="uniformW", n_levels=32, n_max=31)
    source = MPJRDSynapse(cfg, init_n=0)
    source.L.fill_(17)
    source.I.fill_(1.23)
    source.eligibility.fill_(0.77)
    source.protection.fill_(True)
    source.sat_time.fill_(4.0)
    state = source.get_state()

    target = MPJRDSynapse(cfg, init_n=0)
    target.load_state(state)

    assert target.L.item() == source.L.item()
    expected_n = round((source.L.item() / (cfg.n_levels - 1)) * cfg.n_max)
    assert target.N.item() == expected_n
    assert target.W.item() == pytest.approx(source.W.item())
    assert target.I.item() == pytest.approx(source.I.item())
