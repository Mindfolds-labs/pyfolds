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
    def test_stdp_updates_stdp_eligibility_online(self, full_config):
        """Online STDP update deve acumular em stdp_eligibility."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        before = torch.stack([torch.cat([s.stdp_eligibility for s in d.synapses]) for d in neuron.dendrites])

        x = torch.ones(2, full_config.n_dendrites, full_config.n_synapses_per_dendrite)
        post_spike = torch.ones(2)

        neuron._update_stdp_traces(x, post_spike, dt=1.0)

        after = torch.stack([torch.cat([s.stdp_eligibility for s in d.synapses]) for d in neuron.dendrites])
        assert not torch.allclose(after, before)
        assert torch.all(after >= -full_config.max_eligibility)
        assert torch.all(after <= full_config.max_eligibility)

    def test_stdp_input_source_raw_vs_stp(self):
        """raw deve detectar spike pré mesmo quando STP deprime abaixo do limiar."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        cfg_raw = pyfolds.NeuronConfig(
            n_dendrites=1,
            n_synapses_per_dendrite=1,
            stdp_input_source="raw",
            spike_threshold=0.5,
        )
        cfg_stp = pyfolds.NeuronConfig(
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

        cfg_classic = pyfolds.NeuronConfig(
            n_dendrites=1,
            n_synapses_per_dendrite=1,
            ltd_rule="classic",
            plasticity_mode="stdp",
        )
        cfg_current = pyfolds.NeuronConfig(
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
            n.dendrites[0].synapses[0].stdp_eligibility.fill_(0.5)

        x_no_pre = torch.zeros(1, 1, 1)
        before_classic = n_classic.dendrites[0].synapses[0].stdp_eligibility.item()
        before_current = n_current.dendrites[0].synapses[0].stdp_eligibility.item()

        n_classic._update_stdp_traces(x_no_pre, torch.ones(1), dt=1.0)
        n_current._update_stdp_traces(x_no_pre, torch.ones(1), dt=1.0)

        after_classic = n_classic.dendrites[0].synapses[0].stdp_eligibility.item()
        after_current = n_current.dendrites[0].synapses[0].stdp_eligibility.item()

        assert after_classic == before_classic
        assert after_current <= before_current

    def test_stdp_update_is_batch_size_invariant_for_identical_samples(self):
        """Delta sináptico médio não deve escalar linearmente com batch."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        cfg = pyfolds.NeuronConfig(
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

        before1 = n_b1.dendrites[0].synapses[0].stdp_eligibility.item()
        before8 = n_b8.dendrites[0].synapses[0].stdp_eligibility.item()
        n_b1._update_stdp_traces(x1, p1, dt=1.0)
        n_b8._update_stdp_traces(x8, p8, dt=1.0)
        delta1 = n_b1.dendrites[0].synapses[0].stdp_eligibility.item() - before1
        delta8 = n_b8.dendrites[0].synapses[0].stdp_eligibility.item() - before8

        assert delta1 == pytest.approx(delta8, rel=1e-6, abs=1e-6)


def test_stdp_vectorized_synapse_batch_matches_non_vectorized():
    cfg_kwargs = dict(
        n_dendrites=1,
        n_synapses_per_dendrite=3,
        plasticity_mode="stdp",
        spike_threshold=0.0,
        stdp_trace_threshold=0.0,
        device="cpu",
    )
    cfg_vec = pyfolds.NeuronConfig(**cfg_kwargs, use_vectorized_synapses=True)
    cfg_ref = pyfolds.NeuronConfig(**cfg_kwargs, use_vectorized_synapses=False)

    n_vec = pyfolds.MPJRDNeuronAdvanced(cfg_vec)
    n_ref = pyfolds.MPJRDNeuronAdvanced(cfg_ref)

    x = torch.ones(2, 1, 3)
    post = torch.ones(2)
    n_vec._update_stdp_traces(x, post, dt=1.0)
    n_ref._update_stdp_traces(x, post, dt=1.0)

    vec_row = n_vec.dendrites[0].synapse_batch.stdp_eligibility
    ref_row = torch.stack([s.stdp_eligibility.squeeze(0) for s in n_ref.dendrites[0].synapses])
    assert torch.allclose(vec_row, ref_row, atol=1e-6)


def test_stdp_vectorized_path_equivalence_over_multiple_steps():
    """Caminho synapse_batch deve ser funcionalmente equivalente ao fallback."""
    cfg_kwargs = dict(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        plasticity_mode="stdp",
        spike_threshold=0.25,
        stdp_trace_threshold=0.0,
        tau_pre=15.0,
        tau_post=12.0,
        A_plus=0.02,
        A_minus=0.015,
        device="cpu",
    )
    n_vec = pyfolds.MPJRDNeuronAdvanced(pyfolds.NeuronConfig(**cfg_kwargs, use_vectorized_synapses=True))
    n_ref = pyfolds.MPJRDNeuronAdvanced(pyfolds.NeuronConfig(**cfg_kwargs, use_vectorized_synapses=False))

    x_steps = [
        torch.tensor([[[0.0, 0.3, 0.8, 0.1], [0.2, 0.9, 0.0, 0.7]]], dtype=torch.float32),
        torch.tensor([[[0.4, 0.1, 0.6, 0.0], [0.0, 0.2, 1.0, 0.3]]], dtype=torch.float32),
        torch.tensor([[[0.5, 0.5, 0.0, 0.9], [0.1, 0.0, 0.2, 0.8]]], dtype=torch.float32),
    ]
    post_steps = [torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0])]

    for x, post in zip(x_steps, post_steps):
        n_vec._update_stdp_traces(x, post, dt=1.0)
        n_ref._update_stdp_traces(x, post, dt=1.0)

    vec = torch.stack([d.synapse_batch.stdp_eligibility for d in n_vec.dendrites])
    ref = torch.stack([
        torch.stack([s.stdp_eligibility.squeeze(0) for s in d.synapses])
        for d in n_ref.dendrites
    ])

    assert torch.allclose(vec, ref, atol=1e-6, rtol=1e-5)


def test_stdp_synapse_batch_micro_benchmark_smoke():
    """Micro-benchmark comparativo simples entre caminho vetorizado e fallback."""
    import time

    cfg_base = dict(
        n_dendrites=4,
        n_synapses_per_dendrite=32,
        plasticity_mode="stdp",
        spike_threshold=0.3,
        stdp_trace_threshold=0.0,
        device="cpu",
    )
    n_vec = pyfolds.MPJRDNeuronAdvanced(pyfolds.NeuronConfig(**cfg_base, use_vectorized_synapses=True))
    n_ref = pyfolds.MPJRDNeuronAdvanced(pyfolds.NeuronConfig(**cfg_base, use_vectorized_synapses=False))

    x = torch.rand(16, cfg_base["n_dendrites"], cfg_base["n_synapses_per_dendrite"])
    post = (torch.rand(16) > 0.5).float()

    t0 = time.perf_counter()
    for _ in range(200):
        n_vec._update_stdp_traces(x, post, dt=1.0)
    vec_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(200):
        n_ref._update_stdp_traces(x, post, dt=1.0)
    ref_time = time.perf_counter() - t1

    assert vec_time > 0.0
    assert ref_time > 0.0
