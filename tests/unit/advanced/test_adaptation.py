"""Tests for AdaptationMixin."""

import pytest
import torch
import pyfolds


class TestAdaptationMixin:
    """Test adaptation mechanism (SFA)."""
    
    def test_initialization(self, full_config):
        """Test adaptation parameters."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        assert hasattr(neuron, 'adaptation_current')
        assert neuron.adaptation_increment == full_config.adaptation_increment
    
    def test_adaptation_decay(self, full_config):
        """Test adaptation current decay."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        batch_size = 4
        device = torch.device('cpu')
        
        neuron._ensure_adaptation_current(batch_size, device)
        neuron.adaptation_current.fill_(5.0)
        
        u = torch.zeros(batch_size)
        spikes = torch.zeros(batch_size)
        
        import math
        expected_decay = math.exp(-1.0 / neuron.adaptation_tau)
        
        neuron._apply_adaptation(u, spikes, dt=1.0)
        
        assert torch.allclose(
            neuron.adaptation_current,
            torch.ones(batch_size) * 5.0 * expected_decay
        )
    
    def test_adaptation_increment(self, full_config):
        """Test adaptation increment with spikes."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        batch_size = 3
        
        neuron._ensure_adaptation_current(batch_size, torch.device('cpu'))
        neuron.adaptation_current.zero_()
        
        u = torch.zeros(batch_size)
        spikes = torch.tensor([1.0, 0.0, 1.0])
        
        neuron._apply_adaptation(u, spikes, dt=1.0)
        
        expected = torch.tensor([
            neuron.adaptation_increment,
            0.0,
            neuron.adaptation_increment
        ])
        
        assert torch.allclose(neuron.adaptation_current, expected)


    def test_adaptation_respects_string_inference_mode(self, full_config):
        """mode='inference' (str) não deve aplicar adaptação."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        x = torch.ones(2, full_config.n_dendrites, full_config.n_synapses_per_dendrite)

        out = neuron.forward(x, mode='inference')

        assert 'u_eff' in out
        assert 'adaptation_current' not in out


    def test_forward_updates_u_for_downstream_mixins(self, full_config):
        """Adaptação deve atualizar campo `u` para mixins posteriores (e.g., refratário)."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        x = torch.ones(2, full_config.n_dendrites, full_config.n_synapses_per_dendrite)

        out = neuron.forward(x, mode='online')

        assert 'u_raw' in out
        assert 'u_eff' in out
        assert torch.allclose(out['u'], out['u_eff'])


def test_refractory_blocks_adaptation_spikes(full_config):
    cfg = pyfolds.NeuronConfig(
        n_dendrites=1,
        n_synapses_per_dendrite=1,
        t_refrac_abs=2.0,
        theta_init=0.01,
        theta_min=0.001,
        adaptation_enabled=True,
        adaptation_increment=5.0,
        device="cpu",
    )
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)
    for syn in neuron.dendrites[0].synapses:
        syn.N.fill_(cfg.n_max)
    neuron.dendrites[0]._invalidate_cache()
    neuron.u_stp.fill_(1.0)
    neuron.R_stp.fill_(1.0)

    x = torch.ones(1, 1, 1) * 10.0
    out0 = neuron.forward(x, dt=1.0, collect_stats=False)
    out1 = neuron.forward(x, dt=1.0, collect_stats=False)

    assert out0["spikes"].item() == 1.0
    assert out1["refrac_blocked"].item() is True
    assert out1["spikes"].item() == 0.0


def test_adaptation_updates_once_with_final_post_refractory_spikes(full_config):
    cfg = pyfolds.NeuronConfig(
        n_dendrites=1,
        n_synapses_per_dendrite=1,
        t_refrac_abs=2.0,
        theta_init=0.01,
        theta_min=0.001,
        adaptation_enabled=True,
        adaptation_increment=2.0,
        adaptation_tau=1e6,
        device="cpu",
    )
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)
    for syn in neuron.dendrites[0].synapses:
        syn.N.fill_(cfg.n_max)
    neuron.dendrites[0]._invalidate_cache()
    neuron.u_stp.fill_(1.0)
    neuron.R_stp.fill_(1.0)

    x = torch.ones(1, 1, 1) * 10.0
    out0 = neuron.forward(x, dt=1.0, collect_stats=False)
    out1 = neuron.forward(x, dt=1.0, collect_stats=False)

    assert out0["spikes"].item() == 1.0
    assert out1["spikes"].item() == 0.0
    assert neuron.adaptation_current.item() == pytest.approx(2.0, rel=1e-4)


def test_sfa_reduces_real_spike_on_next_step_same_input(full_config):
    cfg = pyfolds.NeuronConfig(
        n_dendrites=1,
        n_synapses_per_dendrite=1,
        t_refrac_abs=0.0,
        t_refrac_rel=0.0,
        theta_init=0.01,
        theta_min=0.001,
        adaptation_enabled=True,
        adaptation_increment=10.0,
        adaptation_tau=1e6,
        device="cpu",
    )
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)
    for syn in neuron.dendrites[0].synapses:
        syn.N.fill_(cfg.n_max)
    neuron.dendrites[0]._invalidate_cache()
    neuron.u_stp.fill_(1.0)
    neuron.R_stp.fill_(1.0)

    x = torch.ones(1, 1, 1) * 10.0
    out0 = neuron.forward(x, dt=1.0, collect_stats=False)
    out1 = neuron.forward(x, dt=1.0, collect_stats=False)

    assert out0["spikes"].item() == 1.0
    assert out1["refrac_blocked"].item() is False
    assert out1["u_eff"].item() < out1["u_raw"].item()
    assert out1["spikes"].item() == 0.0
