"""Tests for BackpropMixin."""

import pytest
import torch
import pyfolds


class TestBackpropMixin:
    """Test dendritic backpropagation."""
    
    def test_initialization(self, full_config):
        """Test backprop parameters."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        assert neuron.backprop_delay == full_config.backprop_delay
        assert neuron.backprop_signal == full_config.backprop_signal
    
    def test_backprop_queue(self, full_config):
        """Test backprop event queue."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        batch_size = 2
        v_dend = torch.randn(batch_size, full_config.n_dendrites)
        
        neuron._schedule_backprop(10.0, v_dend)
        
        assert len(neuron.backprop_queue) == 1
        assert neuron.backprop_queue[0]['time'] == 10.0
    
    def test_dendrite_amplification_decay(self, full_config):
        """Test amplification decay."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        neuron.dendrite_amplification.fill_(0.5)
        
        import math
        expected_decay = math.exp(-1.0 / neuron.backprop_amp_tau)
        
        neuron._process_backprop_queue(1.0)
        
        assert torch.allclose(
            neuron.dendrite_amplification,
            torch.ones_like(neuron.dendrite_amplification) * 0.5 * expected_decay
        )
