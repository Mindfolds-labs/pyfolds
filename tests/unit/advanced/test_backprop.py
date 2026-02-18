"""Tests for BackpropMixin."""

import pytest
import torch
import pyfolds


class TestBackpropMixin:
    """Test dendritic backpropagation."""
    
    def test_initialization(self, full_config):
        """Test backprop parameters."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        assert neuron.backprop_delay == full_config.backprop_delay
        assert neuron.backprop_signal == full_config.backprop_signal
    
    def test_backprop_queue(self, full_config):
        """Test backprop event queue."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        batch_size = 2
        v_dend = torch.randn(batch_size, full_config.n_dendrites)
        
        neuron._schedule_backprop(10.0, v_dend)
        
        assert len(neuron.backprop_queue) == 1
        assert neuron.backprop_queue[0]['time'] == 10.0
    
    def test_dendrite_amplification_decay(self, full_config):
        """Test amplification decay when there are pending events."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        neuron.dendrite_amplification.fill_(0.5)

        # Sem eventos pendentes, não deve aplicar decaimento.
        neuron._process_backprop_queue(1.0)
        assert torch.allclose(
            neuron.dendrite_amplification,
            torch.ones_like(neuron.dendrite_amplification) * 0.5,
        )

        # Com evento pendente, aplica decaimento por tempo desde último processamento.
        v_dend = torch.zeros(2, full_config.n_dendrites)
        neuron._schedule_backprop(100.0, v_dend)

        import math

        expected_decay = math.exp(-(11.0 - 0.0) / neuron.backprop_amp_tau)
        neuron._process_backprop_queue(11.0)

        assert torch.allclose(
            neuron.dendrite_amplification,
            torch.ones_like(neuron.dendrite_amplification) * 0.5 * expected_decay,
            atol=1e-5,
        )

    def test_bap_proportional_uses_dendritic_contribution(self, full_config):
        """Com bap_proportional=True, ganho deve seguir contribuição."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        cfg = pyfolds.MPJRDConfig(
            n_dendrites=full_config.n_dendrites,
            n_synapses_per_dendrite=full_config.n_synapses_per_dendrite,
            bap_proportional=True,
            device="cpu",
        )
        neuron = pyfolds.MPJRDNeuronAdvanced(cfg)

        v_dend = torch.zeros(2, cfg.n_dendrites)
        contribution = torch.zeros(2, cfg.n_dendrites)
        contribution[:, 0] = 1.0

        neuron._schedule_backprop(0.0, v_dend, dend_contribution=contribution)
        neuron._process_backprop_queue(1.0)

        assert neuron.dendrite_amplification[0] > 0
        assert torch.all(neuron.dendrite_amplification[1:] == 0)
