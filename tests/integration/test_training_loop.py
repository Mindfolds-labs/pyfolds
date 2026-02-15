"""Integration test for training loop."""

import pytest
import torch
import pyfolds


@pytest.mark.integration
class TestTrainingLoop:
    """Test complete training loop."""
    
    def test_batch_training(self, small_config):
        """Test batch training loop."""
        neuron = pyfolds.MPJRDNeuron(small_config)
        neuron.set_mode(pyfolds.LearningMode.BATCH)
        
        # Simulate 10 batches
        for epoch in range(10):
            for batch in range(5):
                x = torch.randn(4, 2, 4)
                out = neuron(x, collect_stats=True)
            
            neuron.apply_plasticity(reward=0.5)
            
            # Verify metrics
            metrics = neuron.get_metrics()
            assert 'N_mean' in metrics
