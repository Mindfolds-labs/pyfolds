"""Tests for STDPMixin."""

import pytest
import torch
import pyfolds
from pyfolds.utils.types import LearningMode


class TestSTDPMixin:
    """Test STDP (Spike-Timing Dependent Plasticity)."""
    
    def test_initialization(self, full_config):
        """Test STDP parameters."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        assert neuron.tau_pre == full_config.tau_pre
        assert neuron.tau_post == full_config.tau_post
        assert neuron.A_plus == full_config.A_plus
    
    def test_trace_decay(self, full_config):
        """Test trace decay."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        batch_size = 2
        device = torch.device('cpu')
        D = full_config.n_dendrites
        S = full_config.n_synapses_per_dendrite
        
        neuron._ensure_traces(batch_size, device)
        neuron.trace_pre.fill_(1.0)
        neuron.trace_post.fill_(1.0)
        
        import math
        decay_pre = math.exp(-1.0 / neuron.tau_pre)
        
        x = torch.zeros(batch_size, D, S)
        post_spike = torch.zeros(batch_size)
        
        neuron._update_stdp_traces(x, post_spike, dt=1.0)
        
        assert torch.allclose(
            neuron.trace_pre,
            torch.ones(batch_size, D, S) * decay_pre
        )
    
    def test_pre_spike_updates_trace(self, full_config):
        """Test pre-synaptic spike updates trace."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        batch_size = 2
        device = torch.device('cpu')
        D = full_config.n_dendrites
        S = full_config.n_synapses_per_dendrite
        
        neuron._ensure_traces(batch_size, device)
        neuron.trace_pre.zero_()
        
        x = torch.zeros(batch_size, D, S)
        x[0, 0, 0] = 1.0  # Spike at (sample0, dend0, syn0)
        
        post_spike = torch.zeros(batch_size)
        
        neuron._update_stdp_traces(x, post_spike, dt=1.0)
        
        assert neuron.trace_pre[0, 0, 0].item() > 0
        assert neuron.trace_pre[0, 0, 1].item() == 0
    
    def test_should_apply_stdp(self, full_config):
        """Test STDP application logic."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        
        assert neuron._should_apply_stdp(LearningMode.ONLINE) is True
        assert neuron._should_apply_stdp(LearningMode.INFERENCE) is False
    def test_stdp_updates_underlying_synapses_online(self, full_config):
        """Online STDP update deve persistir nas sinapses reais."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        before = torch.stack([torch.cat([s.I for s in d.synapses]) for d in neuron.dendrites])

        x = torch.ones(2, full_config.n_dendrites, full_config.n_synapses_per_dendrite)
        post_spike = torch.ones(2)

        neuron._update_stdp_traces(x, post_spike, dt=1.0)

        after = torch.stack([torch.cat([s.I for s in d.synapses]) for d in neuron.dendrites])
        assert not torch.allclose(after, before)
        assert torch.all(after >= full_config.i_min)
        assert torch.all(after <= full_config.i_max)
