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
