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

    def test_batch_plasticity_preserves_local_pre_synaptic_rate(self):
        """BATCH deve manter média local por sinapse (sem normalização cruzada)."""
        cfg = pyfolds.MPJRDConfig(
            n_dendrites=1,
            n_synapses_per_dendrite=4,
            defer_updates=True,
            activity_threshold=0.05,
            neuromod_mode="external",
            plastic=True,
            device="cpu",
        )
        neuron = pyfolds.MPJRDNeuron(cfg)
        neuron.set_mode(LearningMode.BATCH)

        captured_pre_rates = []
        original = neuron.dendrites[0].update_synapses_rate_based

        def wrapped(pre_rate, post_rate, R, dt=1.0, mode=None, _orig=original):
            captured_pre_rates.append(pre_rate.detach().clone())
            return _orig(pre_rate, post_rate, R, dt, mode)

        neuron.dendrites[0].update_synapses_rate_based = wrapped

        # Médias por sinapse no batch: [0.10, 0.40, 0.01, 0.80]
        # activity_threshold=0.05 -> terceira sinapse deve ser mascarada para zero.
        x = torch.tensor([
            [[0.10, 0.40, 0.01, 0.80]],
            [[0.10, 0.40, 0.01, 0.80]],
        ])

        neuron.forward(x, reward=1.0, mode=LearningMode.BATCH)
        neuron.apply_plasticity(reward=1.0)

        assert len(captured_pre_rates) == 1
        expected = torch.tensor([0.10, 0.40, 0.00, 0.80])
        assert torch.allclose(captured_pre_rates[0], expected, atol=1e-6)

    @pytest.mark.parametrize("mode", ["online", "batch", "inference"])
    def test_modes(self, small_config, mode):
        """Test learning modes."""
        neuron = pyfolds.MPJRDNeuron(small_config)
        target = LearningMode(mode)
        
        neuron.set_mode(target)
        assert neuron.mode == target
