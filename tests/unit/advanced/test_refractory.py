"""Tests for RefractoryMixin."""

import pytest
import torch
import pyfolds


class TestRefractoryMixin:
    """Test refractory period."""

    def test_initialization(self, full_config):
        """Test refractory parameters."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        assert neuron.t_refrac_abs == full_config.t_refrac_abs
        assert neuron.t_refrac_rel == full_config.t_refrac_rel

    def test_check_refractory_periods(self, full_config):
        """Absolute and relative windows should be blocked with relative boost only in relative."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        batch_size = 3
        
        neuron._ensure_last_spike_time(batch_size, torch.device('cpu'))
        neuron.last_spike_time = torch.tensor([-10.0, 1.0, 3.0])
        
        blocked, theta_boost = neuron._check_refractory_batch(5.0, batch_size)
        
        # -10.0: 15ms ago → not refractory
        # 1.0: 4ms ago → relative refractory (boost de limiar)
        # 3.0: 2ms ago → absoluto (bloqueado)
        assert blocked[2].item() is True
        assert theta_boost[1].item() == neuron.refrac_rel_strength
    
        neuron._ensure_last_spike_time(batch_size=1, device=torch.device("cpu"))
        neuron.last_spike_time = torch.tensor([0.0])

        blocked, theta_boost = neuron._check_refractory_batch(1.0, batch_size=1)
        assert blocked[0].item() is True
        assert theta_boost[0].item() == 0.0

        blocked, theta_boost = neuron._check_refractory_batch(3.0, batch_size=1)
        assert blocked[0].item() is True
        assert theta_boost[0].item() == neuron.refrac_rel_strength

        blocked, theta_boost = neuron._check_refractory_batch(6.0, batch_size=1)
        assert blocked[0].item() is False
        assert theta_boost[0].item() == 0.0

    def test_update_refractory(self, full_config):
        """Test last spike time update."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        batch_size = 3

        neuron._ensure_last_spike_time(batch_size, torch.device('cpu'))
        neuron.last_spike_time.fill_(-1000.0)
        neuron.time_counter.fill_(5.0)

        spikes = torch.tensor([1.0, 0.0, 1.0])
        neuron._update_refractory_batch(spikes, dt=1.0)

        expected = torch.tensor([5.0, -1000.0, 5.0])
        assert torch.allclose(neuron.last_spike_time, expected)
        assert neuron.time_counter.item() == 6.0
