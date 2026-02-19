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

        assert 'u_adapted' not in out
        assert 'adaptation_current' not in out


    def test_forward_updates_u_for_downstream_mixins(self, full_config):
        """Adaptação deve atualizar campo `u` para mixins posteriores (e.g., refratário)."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        x = torch.ones(2, full_config.n_dendrites, full_config.n_synapses_per_dendrite)

        out = neuron.forward(x, mode='online')

        assert 'u_raw' in out
        assert 'u_adapted' in out
        assert torch.allclose(out['u'], out['u_adapted'])
