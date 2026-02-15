"""Tests for HomeostasisController."""

import pytest
import torch
import pyfolds


class TestHomeostasisController:
    """Test homeostasis."""
    
    def test_initialization(self, small_config):
        """Test initialization."""
        from pyfolds.core import HomeostasisController
        
        homeo = HomeostasisController(small_config)
        assert homeo.theta.shape == (1,)
        assert homeo.theta.item() == small_config.theta_init
    
    def test_update(self, small_config):
        """Test update."""
        from pyfolds.core import HomeostasisController
        
        homeo = HomeostasisController(small_config)
        initial = homeo.theta.item()
        
        homeo.update(0.05)  # Below target
        assert homeo.theta.item() < initial


    def test_is_stable_accepts_tolerance(self, small_config):
        """is_stable should accept optional tolerance."""
        from pyfolds.core import HomeostasisController

        homeo = HomeostasisController(small_config)
        homeo.r_hat.fill_(small_config.target_spike_rate + 0.08)

        assert homeo.is_stable(tolerance=0.1) is True
        assert homeo.is_stable(tolerance=0.05) is False
