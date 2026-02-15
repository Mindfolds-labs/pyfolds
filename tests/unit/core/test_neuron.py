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
    
    
    def test_step_alias(self, small_config, batch_size):
        """`step` deve delegar para `forward` preservando a API pública."""
        neuron = pyfolds.MPJRDNeuron(small_config)
        x = torch.randn(batch_size, 2, 4)

        out = neuron.step(x, reward=0.2, dt=1.0)

        assert out['spikes'].shape == (batch_size,)
        assert 'R' in out

    def test_online_plasticity_updates_when_not_deferred(self, small_config):
        """Modo ONLINE deve atualizar N/I quando defer_updates=False."""
        cfg = pyfolds.MPJRDConfig(
            n_dendrites=small_config.n_dendrites,
            n_synapses_per_dendrite=small_config.n_synapses_per_dendrite,
            defer_updates=False,
            i_eta=1.0,
            i_ltp_th=0.01,
            theta_init=0.0,
            theta_min=0.0,
            theta_max=1.0,
            neuromod_mode="external",
            plastic=True,
            device="cpu",
        )
        neuron = pyfolds.MPJRDNeuron(cfg)
        neuron.set_mode(LearningMode.ONLINE)

        n_before = neuron.N.clone()
        x = torch.ones(8, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
        neuron.step(x, reward=1.0, dt=1.0)
        n_after = neuron.N.clone()

        assert torch.any(n_after > n_before)

    @pytest.mark.parametrize("mode", ["online", "batch", "inference"])
    def test_modes(self, small_config, mode):
        """Test learning modes."""
        neuron = pyfolds.MPJRDNeuron(small_config)
        target = LearningMode(mode)
        
        neuron.set_mode(target)
        assert neuron.mode == target
