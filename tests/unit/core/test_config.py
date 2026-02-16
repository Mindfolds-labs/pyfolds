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


    def test_get_decay_rate_validation(self):
        """Tau inválido deve gerar erro explícito."""
        cfg = pyfolds.MPJRDConfig()
        with pytest.raises(ValueError):
            cfg.get_decay_rate(-1.0)


    def test_get_decay_rate_negative_dt_validation(self):
        """dt negativo deve gerar erro explícito."""
        cfg = pyfolds.MPJRDConfig()
        with pytest.raises(ValueError):
            cfg.get_decay_rate(10.0, dt=-0.1)

    def test_numerical_safety_validation(self):
        """Config deve validar parâmetros numéricos críticos."""
        with pytest.raises(ValueError):
            pyfolds.MPJRDConfig(w_scale=0)

        with pytest.raises(ValueError):
            pyfolds.MPJRDConfig(float_precision="float16")
