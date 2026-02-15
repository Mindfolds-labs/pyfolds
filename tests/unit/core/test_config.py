"""Tests for MPJRDConfig."""

import pytest
import torch
import pyfolds


class TestMPJRDConfig:
    """Test configuration class."""
    
    def test_default_values(self):
        """Test default values."""
        cfg = pyfolds.MPJRDConfig()
        assert cfg.n_dendrites == 4
        assert cfg.n_synapses_per_dendrite == 32
    
    def test_custom_values(self):
        """Test custom values."""
        cfg = pyfolds.MPJRDConfig(
            n_dendrites=8,
            n_synapses_per_dendrite=64,
            plastic=False
        )
        assert cfg.n_dendrites == 8
        assert cfg.n_synapses_per_dendrite == 64
        assert cfg.plastic is False
    
    def test_validation(self):
        """Test validation."""
        with pytest.raises(ValueError):
            pyfolds.MPJRDConfig(n_min=10, n_max=5)
