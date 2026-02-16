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

    def test_u_R_optional_state(self, small_config):
        """u/R devem ser opcionais e nunca lançar exceção no acesso."""
        from pyfolds.core import MPJRDDendrite

        dend = MPJRDDendrite(small_config, dendrite_id=0)
        dend._cached_states = {'N': torch.zeros(4, dtype=torch.int32), 'I': torch.zeros(4)}
        dend._cache_invalid = False

        u_state = dend.u
        r_state = dend.R

        assert u_state is None or isinstance(u_state, torch.Tensor)
        assert r_state is None or isinstance(r_state, torch.Tensor)

    def test_update_uses_local_pre_rate_per_synapse(self, small_config):
        """Each synapse must receive its own pre-synaptic rate sample."""
        from pyfolds.core import MPJRDDendrite

        dend = MPJRDDendrite(small_config, dendrite_id=0)

        captured = []
        for syn in dend.synapses:
            original = syn.update

            def wrapped(pre_rate, post_rate, R, dt=1.0, mode=None, _orig=original):
                captured.append(pre_rate.clone())
                return _orig(pre_rate, post_rate, R, dt, mode)

            syn.update = wrapped  # type: ignore[method-assign]

        pre_rate = torch.tensor([0.1, 0.2, 0.3, 0.4])
        post_rate = torch.tensor([0.5])
        R = torch.tensor([0.2])

        dend.update_synapses_rate_based(pre_rate=pre_rate, post_rate=post_rate, R=R)

        assert len(captured) == 4
        for i, pre in enumerate(captured):
            assert pre.shape == (1,)
            assert torch.isclose(pre[0], pre_rate[i])
