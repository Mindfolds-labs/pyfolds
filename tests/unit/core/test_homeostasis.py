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
