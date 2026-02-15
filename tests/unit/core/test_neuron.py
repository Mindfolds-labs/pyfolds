"""Tests for MPJRDNeuron."""

import pytest
import torch
import pyfolds
from pyfolds.utils.types import LearningMode


class TestMPJRDNeuron:
    """Test basic neuron."""
    
    def test_initialization(self, small_config):
        """Test initialization."""
        neuron = pyfolds.MPJRDNeuron(small_config)
        assert neuron.N.shape == (2, 4)
        assert neuron.theta.shape == (1,)
    
    def test_forward(self, small_config, batch_size):
        """Test forward pass."""
        neuron = pyfolds.MPJRDNeuron(small_config)
        x = torch.randn(batch_size, 2, 4)
        
        out = neuron(x)
        
        assert out['spikes'].shape == (batch_size,)
        assert out['u'].shape == (batch_size,)
        assert out['v_dend'].shape == (batch_size, 2)
    
    @pytest.mark.parametrize("mode", ["online", "batch", "inference"])
    def test_modes(self, small_config, mode):
        """Test learning modes."""
        neuron = pyfolds.MPJRDNeuron(small_config)
        target = LearningMode(mode)
        
        neuron.set_mode(target)
        assert neuron.mode == target
