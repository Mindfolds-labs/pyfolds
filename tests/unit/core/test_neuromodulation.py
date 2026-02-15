"""Tests for Neuromodulator."""

import pytest
import torch
import pyfolds


class TestNeuromodulator:
    """Test neuromodulator."""
    
    def test_external_mode(self):
        """Test external mode."""
        cfg = pyfolds.MPJRDConfig(neuromod_mode="external")
        from pyfolds.core import Neuromodulator
        
        mod = Neuromodulator(cfg)
        R = mod(rate=0.1, r_hat=0.1, reward=0.5)
        
        assert R.item() == 0.5
    
    def test_surprise_mode(self):
        """Test surprise mode."""
        cfg = pyfolds.MPJRDConfig(
            neuromod_mode="surprise",
            sup_k=2.0
        )
        from pyfolds.core import Neuromodulator
        
        mod = Neuromodulator(cfg)
        R = mod(rate=0.2, r_hat=0.1)
        
        assert abs(R.item() - 0.2) < 1e-6
