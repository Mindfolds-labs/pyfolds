"""Tests for MPJRDSynapse."""

import pytest
import torch
import pyfolds


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
