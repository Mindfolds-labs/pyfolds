"""Integration tests for advanced neuron."""

import pytest
import torch
import pyfolds


@pytest.mark.integration
class TestAdvancedNeuron:
    """Test advanced neuron with all features."""
    
    def test_full_pipeline(self, full_config, batch_size):
        """Test complete pipeline."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
        
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        x = torch.randn(batch_size, 4, 8)
        
        out = neuron(x)
        assert out['spikes'].shape == (batch_size,)
        
        # Test batch learning
        neuron.set_mode(pyfolds.LearningMode.BATCH)
        for _ in range(5):
            out = neuron(x, collect_stats=True)
        
        neuron.apply_plasticity()
        
        # Test sleep
        neuron.sleep(duration=10.0)

    def test_backprop_disabled_does_not_apply_dendritic_gain(self):
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        cfg = pyfolds.MPJRDConfig(
            n_dendrites=1,
            n_synapses_per_dendrite=1,
            backprop_enabled=False,
            dendrite_integration_mode="wta_hard",
            theta_init=100.0,
            theta_max=200.0,
            device="cpu",
        )
        neuron_amp = pyfolds.MPJRDNeuronAdvanced(cfg)
        neuron_base = pyfolds.MPJRDNeuronAdvanced(cfg)
        for n in (neuron_amp, neuron_base):
            for syn in n.dendrites[0].synapses:
                syn.N.fill_(cfg.n_max)
            n.dendrites[0]._invalidate_cache()
            n.u_stp.fill_(1.0)
            n.R_stp.fill_(1.0)

        x = torch.ones(1, 1, 1)
        neuron_amp.dendrite_amplification.fill_(5.0)
        out_disabled = neuron_amp.forward(x, collect_stats=False, mode=pyfolds.LearningMode.INFERENCE)

        neuron_base.dendrite_amplification.zero_()
        out_base = neuron_base.forward(x, collect_stats=False, mode=pyfolds.LearningMode.INFERENCE)
        assert torch.allclose(out_disabled["v_dend"], out_base["v_dend"])
