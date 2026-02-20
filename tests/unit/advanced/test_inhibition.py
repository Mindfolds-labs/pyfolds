"""Tests for InhibitionLayer."""

import pytest
import torch
import pyfolds


class TestInhibitionLayer:
    """Test inhibition layer."""
    
    def test_initialization(self):
        """Test layer initialization."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        from pyfolds.advanced import InhibitionLayer
        
        layer = InhibitionLayer(
            n_excitatory=10,
            n_inhibitory=3,
            lateral_strength=0.5
        )
        
        assert layer.n_exc == 10
        assert layer.n_inh == 3
        assert layer.W_E2I.shape == (10, 3)
        assert layer.lateral_kernel.shape == (10, 10)
    
    def test_lateral_kernel(self):
        """Test lateral kernel properties."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        from pyfolds.advanced import InhibitionLayer
        
        layer = InhibitionLayer(n_excitatory=5, n_inhibitory=2)
        kernel = layer.lateral_kernel
        
        # Diagonal should be zero (no self-inhibition)
        assert torch.all(torch.diag(kernel) == 0)
        
        # Values between 0 and 1
        assert torch.all(kernel >= 0)
        assert torch.all(kernel <= 1)
    
    def test_forward_feedforward(self):
        """Test feedforward E→I."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        from pyfolds.advanced import InhibitionLayer
        
        layer = InhibitionLayer(n_excitatory=5, n_inhibitory=2)
        batch_size = 3
        exc_spikes = torch.randint(0, 2, (batch_size, 5)).float()
        
        out = layer(exc_spikes)
        
        assert 'inh_spikes' in out
        assert out['inh_potential'].shape == (2,)

    def test_apply_inhibition_accepts_u_values(self):
        """apply_inhibition deve suportar saída de layer com `u_values`."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        from pyfolds.advanced import InhibitionLayer

        layer = InhibitionLayer(n_excitatory=5, n_inhibitory=2)
        exc_output = {
            "spikes": torch.randint(0, 2, (3, 5)).float(),
            "u_values": torch.rand(3, 5),
            "thetas": torch.full((5,), 0.5),
        }
        inh_out = layer(exc_output["spikes"])
        out = layer.apply_inhibition(exc_output, inh_out)
        assert out["spikes"].shape == (3, 5)

    def test_e2i_initialization_is_deterministic(self):
        """E→I should be reproducible across instances."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        from pyfolds.advanced import InhibitionLayer

        layer_a = InhibitionLayer(n_excitatory=10, n_inhibitory=3)
        layer_b = InhibitionLayer(n_excitatory=10, n_inhibitory=3)

        assert torch.allclose(layer_a.W_E2I, layer_b.W_E2I)


class TestInhibitionMixin:
    """Tests for InhibitionMixin guard clauses."""

    def test_forward_requires_initialized_inhibition(self):
        """Forward should fail fast when inhibition layer is missing."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        from pyfolds.advanced.inhibition import InhibitionMixin

        class DummyLayer(InhibitionMixin):
            pass

        layer = DummyLayer()
        with pytest.raises(RuntimeError, match="Inibição não foi inicializada"):
            layer.forward(torch.zeros(1, 2, 3))
