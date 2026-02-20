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

    def test_stdp_input_source_raw_vs_stp(self):
        """raw deve detectar spike pré mesmo quando STP deprime abaixo do limiar."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        cfg_raw = pyfolds.MPJRDConfig(
            n_dendrites=1,
            n_synapses_per_dendrite=1,
            stdp_input_source="raw",
            spike_threshold=0.5,
        )
        cfg_stp = pyfolds.MPJRDConfig(
            n_dendrites=1,
            n_synapses_per_dendrite=1,
            stdp_input_source="stp",
            spike_threshold=0.5,
        )

        n_raw = pyfolds.MPJRDNeuronAdvanced(cfg_raw)
        n_stp = pyfolds.MPJRDNeuronAdvanced(cfg_stp)

        for n in (n_raw, n_stp):
            n.u_stp.fill_(0.1)
            n.R_stp.fill_(0.1)

        x = torch.ones(1, 1, 1)
        n_raw.forward(x, mode=LearningMode.ONLINE)
        n_stp.forward(x, mode=LearningMode.ONLINE)

        assert n_raw.trace_pre[0, 0, 0].item() > 0.0
        assert n_stp.trace_pre[0, 0, 0].item() == 0.0

    def test_ltd_rule_classic_vs_current(self):
        """classic usa pre_spike; current preserva regra legada dependente de post."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        cfg_classic = pyfolds.MPJRDConfig(
            n_dendrites=1,
            n_synapses_per_dendrite=1,
            ltd_rule="classic",
            plasticity_mode="stdp",
        )
        cfg_current = pyfolds.MPJRDConfig(
            n_dendrites=1,
            n_synapses_per_dendrite=1,
            ltd_rule="current",
            plasticity_mode="stdp",
        )
        n_classic = pyfolds.MPJRDNeuronAdvanced(cfg_classic)
        n_current = pyfolds.MPJRDNeuronAdvanced(cfg_current)

        for n in (n_classic, n_current):
            n._ensure_traces(1, torch.device("cpu"))
            n.trace_post.fill_(1.0)
            n.trace_pre.zero_()
            n.dendrites[0].synapses[0].I.fill_(0.5)

        x_no_pre = torch.zeros(1, 1, 1)
        before_classic = n_classic.dendrites[0].synapses[0].I.item()
        before_current = n_current.dendrites[0].synapses[0].I.item()

        n_classic._update_stdp_traces(x_no_pre, torch.ones(1), dt=1.0)
        n_current._update_stdp_traces(x_no_pre, torch.ones(1), dt=1.0)

        after_classic = n_classic.dendrites[0].synapses[0].I.item()
        after_current = n_current.dendrites[0].synapses[0].I.item()

        assert after_classic == before_classic
        assert after_current <= before_current

    def test_stdp_update_is_batch_size_invariant_for_identical_samples(self):
        """Delta sináptico médio não deve escalar linearmente com batch."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        cfg = pyfolds.MPJRDConfig(
            n_dendrites=1,
            n_synapses_per_dendrite=1,
            plasticity_mode="stdp",
            i_min=-10.0,
            i_max=10.0,
            spike_threshold=0.0,
            stdp_trace_threshold=0.0,
        )

        n_b1 = pyfolds.MPJRDNeuronAdvanced(cfg)
        n_b8 = pyfolds.MPJRDNeuronAdvanced(cfg)

        x1 = torch.ones(1, 1, 1)
        p1 = torch.ones(1)
        x8 = torch.ones(8, 1, 1)
        p8 = torch.ones(8)

        before1 = n_b1.dendrites[0].synapses[0].I.item()
        before8 = n_b8.dendrites[0].synapses[0].I.item()
        n_b1._update_stdp_traces(x1, p1, dt=1.0)
        n_b8._update_stdp_traces(x8, p8, dt=1.0)
        delta1 = n_b1.dendrites[0].synapses[0].I.item() - before1
        delta8 = n_b8.dendrites[0].synapses[0].I.item() - before8

        assert delta1 == pytest.approx(delta8, rel=1e-6, abs=1e-6)
