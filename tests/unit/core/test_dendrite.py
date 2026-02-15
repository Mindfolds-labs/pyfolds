"""Tests for MPJRDDendrite."""

import pytest
import torch
import pyfolds


class TestMPJRDDendrite:
    """Test dendrite."""
    
    def test_initialization(self, small_config):
        """Test initialization."""
        from pyfolds.core import MPJRDDendrite
        
        dend = MPJRDDendrite(small_config, dendrite_id=0)
        assert dend.n_synapses == 4
        assert dend.id == 0
    
    def test_forward(self, small_config, batch_size):
        """Test forward pass."""
        from pyfolds.core import MPJRDDendrite
        
        dend = MPJRDDendrite(small_config, dendrite_id=0)
        x = torch.randn(batch_size, 4)
        
        v = dend(x)
        assert v.shape == (batch_size,)
